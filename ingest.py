import os
import concurrent.futures
from datetime import datetime
import logging
import tempfile
import time
import boto3
import botocore
import click
import numpy as np
import json
import psycopg
import faiss
import mercantile

logging.getLogger().setLevel(logging.INFO)

PCA_TRAINING_SAMPLE_SIZE_PERCENT = 0.10
bucket = boto3.resource("s3").Bucket(os.environ["BUCKET_NAME"])


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


def fetch_data_from_s3():
    logging.info("Fetching data from S3. Searching for embeddings files under prefix: demo/data")
    
    embeddings_files  = [o.key for o in bucket.objects.filter(Prefix="demo/data") if o.key.endswith("embeddings.fvecs")] 

    logging.info(f"Found {len(embeddings_files)} embeddings_files to retrieve")
    
    def fetch_embeddings(s3_key): 
        with tempfile.NamedTemporaryFile() as tmpfile: 
            bucket.Object(s3_key).download_file(tmpfile.name)
            embeddings = fvecs_read(tmpfile.name)
        metadata = json.loads(
            bucket.Object(s3_key.replace("embeddings.fvecs", "metadata.json")).get()["Body"].read()
        )
    
        assert len(embeddings) == len(metadata)  
        return embeddings, metadata
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exec: 
        embeddings_metadata_subsets = list(exec.map(
            lambda f: fetch_embeddings(f),
            embeddings_files
        ))  
    
    return embeddings_metadata_subsets

def get_pca_matrix(retrain=False): 
    
    logging.info("Fetching PCA matrix")

    if not retrain:
        logging.info("Retrain parameter is false, searching for vector transform file (demo/data/PCA128_IVF1024_Flat.index) in S3.")    
        try: 
            with tempfile.NamedTemporaryFile() as tmpfile:
                bucket.Object("demo/data/512_to_128_pca_matrix.pca").download_file(tmpfile.name)

                logging.info("Vector transform file found! Loading from S3 ")

                return faiss.read_VectorTransform(tmpfile.name)
            
        except botocore.exceptions.ClientError as e:
            if int(e.response['Error'] ['Code']) != 404:
                raise e
            logging.info("Vector transform file not found. Proceeding to training step.")
            
    embeddings_metadata_batches = fetch_data_from_s3()
    
    embeddings = []
    for _embeddings, _ in embeddings_metadata_batches: 
        embeddings.extend(_embeddings)
    
    logging.info("Converting embeddings to np.array")
    embeddings = np.array(embeddings)    
        
    logging.info(f"Training PCA with training set size: {PCA_TRAINING_SAMPLE_SIZE_PERCENT*100} percent of total dataset ")
    
    rand_indexes = np.random.randint(
        low=0, high=len(embeddings), size=int(PCA_TRAINING_SAMPLE_SIZE_PERCENT * len(embeddings))
    )
    pca_training_subset = embeddings[rand_indexes]
    
    # define PCA matrix
    pca_matrix = faiss.PCAMatrix(512, 128)
    
    logging.info("Training PCA")
    # train
    pca_matrix.train(pca_training_subset)

    with tempfile.NamedTemporaryFile() as tmpfile:
        faiss.write_VectorTransform(pca_matrix, tmpfile.name)
        bucket.Object("demo/data/512_to_128_pca_matrix.pca").upload_file(tmpfile.name)

    logging.info("Done")
        
    return pca_matrix

@click.command()
@click.option('--reingest-data', default=False)
@click.option('--retrain-index', default=False)
def ingest(reingest_data: bool=False, retrain_index:bool=False): 
    
    logging.info(f"Running ingestion with parameters: reingest-data={reingest_data} and retrain-index={retrain_index}")

    pca_matrix = get_pca_matrix()

    database_secrets = {k:os.environ[k] for k in ["host", "dbname", "user", "password", "port"]}
    
    embeddings_metadata_batches = fetch_data_from_s3()
    
    with psycopg.connect(**database_secrets) as conn:
        
        with conn.cursor() as cursor:     
            
            logging.info("Creating pgVectors extension (if not exits)")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            logging.info("Creating table (if not exists)")
            cursor.execute("CREATE TABLE IF NOT EXISTS images (id bigserial PRIMARY KEY, embedding vector(128), datetime timestamp, quadkey VARCHAR(16))")
            
            records_count = cursor.execute("SELECT COUNT(id) from images").fetchone()[0]

            logging.info(f"Found {records_count} records in database")
            
            if records_count == 0 or reingest_data: 
                
                logging.info("Dropping index from table (if exists)")
                cursor.execute("DROP INDEX IF EXISTS embeddings_ivfflat_l2_3186")
                
                logging.info("Truncating table")
                cursor.execute("TRUNCATE images")

                for i, (embeddings, metadata) in enumerate(embeddings_metadata_batches): 
                    
                    with cursor.copy("COPY images (embedding, datetime, quadkey) FROM STDIN") as copy:    
                        
                        logging.info(f"Batch {i} out of {len(embeddings_metadata_batches)}: applying PCA matrix")
                        
                        reduced_embeddings = pca_matrix.apply(embeddings)
                        
                        logging.info(f"Batch {i} out of {len(embeddings_metadata_batches)}: loading records in database")

                        for re, m in zip(reduced_embeddings, metadata):
                            record = (
                                str(re.tolist()), 
                                datetime.strptime(m["date"], "%Y-%m-%d"), 
                                mercantile.quadkey(mercantile.Tile(**m["tile"]))
                            )
                            copy.write_row(record)

                    
                logging.info("Creating index")
                
                cursor.execute("SET maintenance_work_mem='2060MB'")
                cursor.execute("CREATE INDEX IF NOT EXISTS embeddings_ivfflat_l2_3186 ON images USING ivfflat (embedding vector_l2_ops) WITH (lists = 3186);")

                ## The following code checks the status of the index creation, however it needs to be 
                ## executed in another thread, since the `CREATE INDEX` command above blocks this thread
                # tuples_done = tuples_total = 0

                # while tuples_done == 0 or tuples_done<tuples_total: 
                #     phase, tuples_done, tuples_total = cursor.execute("SELECT phase, tuples_done, tuples_total FROM pg_stat_progress_create_index").fetchone()
                #     logging.info(f"Index building phase: {phase}, tuples done: {tuples_done} out of total: {tuples_total}")
                #     time.sleep(60)

                cursor.execute("RESET maintenance_work_mem")

            # check that re-ingest data isn't set to true to avoid re-indexeing a brand new index
            if retrain_index and not reingest_data: 

                logging.info("Re-training index")

                cursor.execute("SET maintenance_work_mem='2060MB'")
                cursor.execute("REINDEX INDEX embeddings_ivfflat_l2_3186")
                cursor.execute("RESET maintenance_work_mem")

    logging.info("Ingestion done!")


if __name__ == "__main__": 
    ingest()
    