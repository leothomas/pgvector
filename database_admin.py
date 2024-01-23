# standard lib imports
from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import faiss
import numpy as np
import sqlalchemy
import vecs
from sqlalchemy.dialects import postgresql
import mercantile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class PGVectorClient:
    def __init__(
        self,
        collection_name: str,
        database_connection_string: str,
        vector_dimensions: int,
        index_type: str,
        index_build_parameters: Dict = {},
        distance_metric: str = "EUCLIDEAN",
        overwrite_if_exists: bool = False,
    ):
        if index_type not in ["HNSW", "IVF"]:
            raise Exception("Index must be one of HNSW, IVF")
        self.index_type = (
            vecs.IndexMethod.hnsw if index_type == "HSNW" else vecs.IndexMethod.ivfflat
        )
        self.index_build_parameters = {}
        if index_build_parameters:
            self.index_build_parameters = (
                vecs.IndexArgsHNSW(**index_build_parameters)
                if index_type == vecs.IndexMethod.hnsw
                else vecs.IndexArgsIVFFlat(**index_build_parameters)
            )

        if distance_metric not in ["EUCLIDEAN", "COSINE"]:
            raise Exception("Distance metric must be one of EUCLIDEAN, COSINE")
        self.distance_metric = (
            vecs.IndexMeasure.cosine_distance
            if distance_metric == "COSINE"
            else vecs.IndexMeasure.l2_distance
        )
        self.vecs_client = vecs.create_client(database_connection_string)

        # Bump maintenance memory to train index on larger file set
        with self.vecs_client.Session() as session:
            with session.begin():
                session.execute(sqlalchemy.text("SET maintenance_work_mem = '2048MB';"))
                session.execute(sqlalchemy.text("SET work_mem = '128MB';"))
                session.execute(sqlalchemy.text("SET random_page_cost = '1.1';"))
                session.execute(sqlalchemy.text("SET effective_io_concurrency = 100;"))
                session.execute(sqlalchemy.text("SET temp_buffers = '512MB';"))

        self.metadata_table = self.create_metadata_table()
        self.collection = self.create_collection(
            collection_name,
            vector_dimensions,
        )

        if self.get_num_records() > 0 and overwrite_if_exists:
            with self.vecs_client.Session() as session:
                with session.begin():
                    session.execute(
                        sqlalchemy.text(
                            f"TRUNCATE TABLE vecs.metadata, vecs.{collection_name};"
                        )
                    )

        logger.info(
            f"Created metadata table and collection with name "
            f"{collection_name} and vector dimensions {vector_dimensions}",
        )

    def create_metadata_table(self):
        metadata_table = sqlalchemy.Table(
            "metadata",
            self.vecs_client.meta,
            sqlalchemy.Column("key", sqlalchemy.String, primary_key=True),
            sqlalchemy.Column("type", sqlalchemy.String),
            sqlalchemy.Column(
                "value",
                postgresql.JSONB,
                server_default=sqlalchemy.text("'{}'::jsonb"),
                nullable=False,
            ),
            sqlalchemy.Column(
                "timestamp",
                sqlalchemy.DateTime,
                default=datetime.utcnow,
            ),
            extend_existing=True,
        )

        # if not sqlalchemy.inspect(self.vecs_client.engine).has_table("metadata"):
        metadata_table.create(self.vecs_client.engine, checkfirst=True)

        return metadata_table

    def create_collection(self, collection_name: str, vector_dimension: int):
        return self.vecs_client.get_or_create_collection(
            name=collection_name,
            dimension=vector_dimension,
        )

    def train_pca_matrix(self, vectors: np.ndarray):
        # define PCA matrix
        pca_matrix = faiss.PCAMatrix(
            vectors.shape[1],
            self.collection.dimension,
        )
        # train
        pca_matrix.train(vectors)

        A = faiss.vector_to_array(pca_matrix.A).reshape(
            pca_matrix.d_out,
            pca_matrix.d_in,
        )
        b = faiss.vector_to_array(pca_matrix.b)

        with self.vecs_client.Session() as session:
            with session.begin():
                values_to_insert = [
                    ("pca.A", A.tolist(), "numpy.ndarray"),
                    ("pca.b", b.tolist(), "numpy.ndarray"),
                    ("d_in", pca_matrix.d_in, "int"),
                    ("d_out", pca_matrix.d_out, "int"),
                ]

                for key, data, type in values_to_insert:
                    _values = dict(
                        key=key,
                        value=json.dumps({"data": data}),
                        type=type,
                    )

                    insert_statement = postgresql.insert(self.metadata_table).values(
                        **_values,
                    )

                    upsert_statement = insert_statement.on_conflict_do_update(
                        constraint="metadata_pkey",
                        set_=_values,
                    )

                    session.execute(upsert_statement)

        logger.info(
            f"Trained and saved PCA dimensionality reduction matrix. Dim: {vectors.shape[1]} to {self.collection.dimension}",
        )

    @lru_cache
    def fetch_pca_matrix_components(self):
        with self.vecs_client.Session() as session:
            with session.begin():
                statement = sqlalchemy.select(self.metadata_table.columns.value).where(
                    self.metadata_table.columns.key == "pca.A",
                )
                A = np.array(
                    json.loads(
                        session.execute(statement).one()[0],
                    )["data"],
                )

                statement = sqlalchemy.select(self.metadata_table.columns.value).where(
                    self.metadata_table.columns.key == "pca.b",
                )
                b = np.array(
                    json.loads(
                        session.execute(statement).one()[0],
                    )["data"],
                )

        return A, b

    def apply_pca_dimensionality_reduction(self, vectors: np.ndarray):
        A, b = self.fetch_pca_matrix_components()

        # wrap single vector in a 2d array, apply transformation
        # and return single vector
        if vectors.ndim == 1:
            return (np.array([vectors]) @ A.T + b)[0]

        # apply transformation to 2d array
        return vectors @ A.T + b

    def train_scalar_quantization(self, vectors: np.ndarray):
        # Add min and max value across all vectors in order to apply scalar
        # quantization to search vectors in future
        with self.vecs_client.Session() as session:
            with session.begin():
                values_to_insert = [
                    ("sq.min", vectors.min().astype(float), "numpy.float32"),
                    ("sq.max", vectors.max().astype(float), "numpy.float32"),
                ]
                for key, data, type in values_to_insert:
                    _values = dict(
                        key=key,
                        value=json.dumps({"data": data}),
                        type=type,
                    )

                    insert_statement = postgresql.insert(self.metadata_table).values(
                        **_values,
                    )

                    upsert_statement = insert_statement.on_conflict_do_update(
                        constraint="metadata_pkey",
                        set_=_values,
                    )

                    session.execute(upsert_statement)

    @lru_cache
    def fetch_sq_min_max(self):
        with self.vecs_client.Session() as session:
            with session.begin():
                statement = sqlalchemy.select(self.metadata_table.columns.value).where(
                    self.metadata_table.columns.key == "sq.min",
                )
                _min = np.float32(
                    json.loads(session.execute(statement).one()[0])["data"],
                )

                statement = sqlalchemy.select(self.metadata_table.columns.value).where(
                    self.metadata_table.columns.key == "sq.max",
                )
                _max = np.float32(
                    json.loads(session.execute(statement).one()[0])["data"],
                )

        return _min, _max

    def apply_scalar_quantization(self, vectors: np.ndarray):
        _min, _max = self.fetch_sq_min_max()
        normalized_vectors = np.clip((vectors - _min) / (_max - _min), 0, 1)
        return (normalized_vectors * 256).astype(np.uint8)

    def load(self, data: List[Dict]) -> None:
        if data[0]["embedding"].shape[0] < self.collection.dimension:
            raise Exception(
                "Cannot index vectors with lower dimensionality than" "collection",
            )

        if data[0]["embedding"].shape[0] > self.collection.dimension:
            reduced_embeddings = self.apply_pca_dimensionality_reduction(
                np.array([d["embedding"] for d in data])
            )

            data = [
                {
                    "id": d["id"],
                    "quadkey": d["quadkey"],
                    "date": d["image_dt"].strftime("%Y-%m-%d"),
                    "timestamp": int(d["image_dt"].timestamp()),
                    "min_lon": mercantile.bounds(
                        mercantile.quadkey_to_tile(d["quadkey"])
                    ).west,
                    "min_lat": mercantile.bounds(
                        mercantile.quadkey_to_tile(d["quadkey"])
                    ).south,
                    "max_lon": mercantile.bounds(
                        mercantile.quadkey_to_tile(d["quadkey"])
                    ).east,
                    "max_lat": mercantile.bounds(
                        mercantile.quadkey_to_tile(d["quadkey"])
                    ).north,
                    "embedding": reduced_embedding,
                }
                for reduced_embedding, d in zip(reduced_embeddings, data)
            ]

        n_records = self.get_num_records()

        # logger.info(f"Table currently loaded with {n_records} records")
        # Records to insert are expected to be formatted:
        # Tuple[
        #   id: Union[int,str],
        #   embedding: List[float],
        #   metadata:Dict[str, Any]
        # ]
        # TODO: let postgres generate ids using serial
        self.collection.upsert(
            [
                (
                    # n_records + i,
                    d["id"],
                    d["embedding"],
                    {k: v for k, v in d.items() if k not in ["embedding", "id"]},
                )
                for i, d in enumerate(data)
            ],
        )

        n_records = self.get_num_records()

        logger.info(
            f"Done inserting records. Table now loaded with {n_records} records",
        )

        return

    def train_index(self) -> None:
        # TODO: add supervision for index creation status
        _params = dict(
            measure=self.distance_metric,
            method=self.index_type,
        )
        if self.index_build_parameters:
            _params["index_arguments"] = self.index_build_parameters

        logger.info(
            f"Initiating index training process with parameters: {_params}",
        )

        self.collection.create_index(**_params)
        return

    def check_index_progress(self):
        with self.vecs_client.Session() as session:
            with session.begin():
                _t = (
                    "tuples_done"
                    if self.index_type == vecs.IndexMethod.ivfflat
                    else "blocks_done"
                )
                return session.execute(
                    sqlalchemy.text(
                        f'SELECT phase, round(100.0 * {_t} / nullif({_t}, 0), 1) AS "%" FROM pg_stat_progress_create_index',
                    ),
                ).fetchone()

    def get_num_records(self) -> int:
        with self.vecs_client.Session() as session:
            with session.begin():
                return session.execute(
                    sqlalchemy.func.count(self.collection.table.c.id),
                ).scalar()

    def query(
        self,
        vector: np.ndarray,
        neighbors: Optional[int] = None,
        distance: Optional[Union[int, float]] = None,
        filters: Optional[Dict] = None,
        nprobe: Optional[int] = 32,
    ) -> List:
        if len(vector) < self.collection.dimension:
            raise Exception(
                "Cannot search using an input vector with with smaller"
                "dimensionality than collection ",
            )
        if len(vector) > self.collection.dimension:
            vector = self.apply_pca_dimensionality_reduction(vector)

        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        query_params = dict(
            data=vector,
            limit=neighbors,
            distance=distance,
            measure=self.distance_metric,
            include_value=True,
            include_metadata=True,
            probes=nprobe,
        )
        if filters:
            query_params["filters"] = filters

        # NOTE: vecs only exposes n-neighbors query (not distance)
        return self.collection.query(**query_params)
