import argparse
import os
import sys
import json
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, SRC_DIR)

from vrag.directory import RESULT_DIR


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True)
    args = argparser.parse_args()
    return args


def get_embedding_similarity(query_embeddings, db_embedding: np.ndarray):
    assert (len(db_embedding.shape) == 1) and (len(query_embeddings.shape) == 2)
    db_embedding = np.expand_dims(db_embedding, axis=0)
    similarities = cosine_similarity(query_embeddings, db_embedding).reshape(-1)
    assert len(similarities) == len(query_embeddings)
    return similarities


def create_embeddings(config: str):
    """
    Creates video embeddings from R-MAC features

    Args:
        config: configuration file containing hyperparameters
    """
    import data.datasets.evve as dset
    query_video_ids = dset.get_query_video_ids()
    db_video_ids = dset.get_db_video_ids()

    result_dir = os.path.join(RESULT_DIR, config, 'evve')
    similarity_file = os.path.join(result_dir, 'aqe-similarities.json')
    similarity_dict = {query_id: {} for query_id in query_video_ids}
    embedding_dir = os.path.join(result_dir, 'embeddings')
    query_embeddings = np.array([np.load(os.path.join(embedding_dir, video_id + '.npy'))
                                 for video_id in query_video_ids])
    db_embeddings = np.array([np.load(os.path.join(embedding_dir, video_id + '.npy'))
                              for video_id in db_video_ids])
    query_embeddings = get_aqe_embeddings(query_embeddings=query_embeddings, db_embeddings=db_embeddings)

    print('INFO: computing similarities for EVVE AQE embeddings...')
    for db_id in tqdm(db_video_ids):
        db_embedding = np.load(os.path.join(embedding_dir, db_id + '.npy'))
        similarities = get_embedding_similarity(query_embeddings=query_embeddings, db_embedding=db_embedding)
        for i, query_id in enumerate(query_video_ids):
            similarity_dict[query_id][db_id] = float(similarities[i])

    with open(similarity_file, 'w') as f:
        json.dump(similarity_dict, f)

    print('INFO: evaluating similarities for EVVE+AQE!')
    dset.evaluate(video_similarities=similarity_dict)


def get_aqe_embeddings(query_embeddings: np.ndarray, db_embeddings, num_neighbors: int = 10):
    norm_query_embeddings = normalize(query_embeddings)
    norm_db_embeddings = normalize(db_embeddings)

    similarity_mtx = cosine_similarity(norm_query_embeddings, norm_db_embeddings)
    neighbor_idxs = np.argsort(similarity_mtx, axis=-1)[:, -num_neighbors:].reshape(-1)

    num_queries, num_dims = query_embeddings.shape
    neighbor_embeddings = norm_db_embeddings[neighbor_idxs, :].reshape(num_queries, num_neighbors, num_dims)
    updated_query_embeddings = (norm_query_embeddings + np.sum(neighbor_embeddings, axis=1)) / (num_neighbors + 1)
    return updated_query_embeddings


def main():
    args = _parse_arguments()
    create_embeddings(config=args.config)


if __name__ == '__main__':
    main()
