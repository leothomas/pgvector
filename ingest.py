import os
import concurrent.futures
import logging
import tempfile
import boto3
import numpy as np
import psycopg
from psycopg.rows import dict_row
from typing import List
from database_admin import PGVectorClient

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.ERROR)


PCA_TRAINING_SAMPLE_SIZE_PERCENT = 0.001
POSTGRES_BATCH_SIZE = 1000

bucket = boto3.resource("s3").Bucket(os.environ["BUCKET_NAME"])
dbs = {k: os.environ[k] for k in ["host", "dbname", "user", "password", "port"]}

pgvector_client = PGVectorClient(
    collection_name="similaritysearch",
    database_connection_string=f"postgresql://{dbs['user']}:{dbs['password']}@{dbs['host']}:{dbs['port']}/{dbs['dbname']}",
    vector_dimensions=128,
    index_type="IVF",
    distance_metric="EUCLIDEAN",
    overwrite_if_exists=True,
)


def ivecs_write(fname, m):
    """Writes a numpy array (vector) of integers to disk in np.int32 format."""
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype="int32")
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    """Writes a numpy array (vector) of integers to disk in np.float32 format."""
    m = m.astype("float32")
    ivecs_write(fname, m.view("int32"))


def ivecs_read(fname):
    """Reads from an .ivecs file into a numpy array of type np.int32"""
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    """Reads from an .fvecs file into a numpy array of type np.float32"""
    return ivecs_read(fname).view("float32")


def fetch(
    s3_key: str,
    with_metadata: bool = False,
    load_percentage: float = 1.0,
):
    logger.info(
        f"Fetching embeddings file {s3_key} {'with metadata' if with_metadata else ''}"
    )
    with tempfile.NamedTemporaryFile() as tmpfile:
        bucket.Object(s3_key).download_file(tmpfile.name)
        embeddings = fvecs_read(tmpfile.name)

    # Note: load_perecentage is not applied to metadata
    # since random subsample should just be applied to
    # embeddings when training the PCA matrix
    rand_indexes = np.random.choice(
        embeddings.shape[0], size=int(len(embeddings) * load_percentage)
    )
    if not with_metadata:
        return [{"embedding": e} for e in embeddings[rand_indexes]]

    logger.info(f"Fetching ids for embeddings file {s3_key}, to look up in database")
    with tempfile.NamedTemporaryFile() as tmpfile:
        bucket.Object(
            s3_key.replace("embeddings/", "ids/").replace(".fvecs", ".txt")
        ).download_file(tmpfile.name)

        with open(tmpfile.name, "r") as file:
            ids = [int(l) for l in file.readlines()]

    logger.info(f"Fetching metadata from RDS for {len(ids)} ids")

    with psycopg.connect(
        conninfo=os.environ["PROD_DB_CONNECTION_STRING"],
        row_factory=dict_row,
    ) as conn:
        # lookup metadata for each ids in batches of 1000
        metadata = []
        for i in range(0, len(ids), POSTGRES_BATCH_SIZE):
            result = conn.execute(
                f"SELECT * FROM images WHERE id IN {tuple(ids[i:i+POSTGRES_BATCH_SIZE])}"
            )
            metadata.extend(result.fetchall())

    logger.info(f"Sample metadata: {metadata[0]}")

    assert len(embeddings) == len(metadata)

    return [
        {"embedding": embedding, **_metadata}
        for embedding, _metadata in zip(embeddings, metadata)
    ]


if __name__ == "__main__":
    embeddings_files = [
        o.key
        for o in bucket.objects.filter(Prefix="embeddings")
        if o.key.endswith(".fvecs")
    ][:5]

    logging.info(f"Found {len(embeddings_files)} embeddings files to process")

    pca_training_emebddings_files = np.array(embeddings_files)[
        np.random.choice(len(embeddings_files), size=25)
    ]
    logger.info(
        f"Fetching embeddings for PCA training from {len(pca_training_emebddings_files)} files"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exec:
        data_subsets = list(
            exec.map(
                lambda f: fetch(f, load_percentage=0.01),
                pca_training_emebddings_files,
            )
        )

    pca_training_vectors = np.array(
        [d["embedding"] for subset in data_subsets for d in subset]
    )

    logger.info(f"Training PCA matrix with {pca_training_vectors.shape} vectors")
    pgvector_client.train_pca_matrix(pca_training_vectors)

    logger.info(
        f"PCA matrix trained. Loading data from {len(embeddings_files)} files into vector database",
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exec:
        result = list(
            exec.map(
                lambda f: pgvector_client.load(fetch(f, with_metadata=True)),
                embeddings_files,
            )
        )

    logger.info("Data inserted. Training index")
    pgvector_client.train_index()

    logger.info(
        f"Done training index. Instance loaded with: {pgvector_client.get_num_records()} records.",
    )

    # search_embedding = data[0]["embedding"]
