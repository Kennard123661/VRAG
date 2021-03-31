import argparse
import os
import sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


if __name__ == '__main__':
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, SRC_DIR)

from vrag.directory import RESULT_DIR


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True)
    argparser.add_argument('--dataset', type=str, choices=['ccweb', 'evve', 'fivr5k', 'fivr200k'], required=True)
    args = argparser.parse_args()
    return args


def get_embedding_similarity(query_embeddings, db_embedding: np.ndarray):
    assert (len(db_embedding.shape) == 1) and (len(query_embeddings.shape) == 2)
    db_embedding = np.expand_dims(db_embedding, axis=0)
    similarities = cosine_similarity(query_embeddings, db_embedding).reshape(-1)
    assert len(similarities) == len(query_embeddings)
    return similarities


def create_embeddings(dataset: str, config: str):
    """
    Creates video embeddings from R-MAC features

    Args:
        dataset: dataset to generate video embeddings
        config: configuration file containing hyperparameters
        device: device id
        checkpoint: model checkpoint name
        use_aqe: set to True to use Average Query Expansion
    """
    if dataset == 'ccweb':
        import data.datasets.ccweb_video as dset
        query_video_ids, db_video_ids = dset.get_query_db_videos()
    elif dataset == 'evve':
        import data.datasets.evve as dset
        query_video_ids = dset.get_query_video_ids()
        db_video_ids = dset.get_db_video_ids()
    elif dataset == 'fivr5k':
        import data.datasets.fivr5k as dset
        query_video_ids = dset.get_query_video_ids()
        db_video_ids = dset.get_db_video_ids()
    elif dataset == 'fivr200k':
        import data.datasets.fivr200k as dset
        query_video_ids = dset.get_query_video_ids()
        db_video_ids = dset.get_db_video_ids()
    else:
        raise NotImplementedError
    similarity_dict = {query_id: {} for query_id in query_video_ids}
    result_dir = os.path.join(RESULT_DIR, config, dataset)
    similarity_file = os.path.join(result_dir, 'similarities.json')
    embedding_dir = os.path.join(result_dir, 'embeddings')
    query_embeddings = np.array([np.load(os.path.join(embedding_dir, video_id + '.npy'))
                                 for video_id in query_video_ids])

    print('INFO: computing similarities for {} embeddings...'.format(dataset.upper()))
    for db_id in tqdm(db_video_ids):
        db_embedding = np.load(os.path.join(embedding_dir, db_id + '.npy'))
        similarities = get_embedding_similarity(query_embeddings=query_embeddings, db_embedding=db_embedding)
        for i, query_id in enumerate(query_video_ids):
            similarity_dict[query_id][db_id] = float(similarities[i])

    with open(similarity_file, 'w') as f:
        json.dump(similarity_dict, f)

    print('INFO: evaluating similarities for {}!'.format(dataset.upper()))
    dset.evaluate(video_similarities=similarity_dict)


def main():
    args = _parse_arguments()
    create_embeddings(dataset=args.dataset, config=args.config)


if __name__ == '__main__':
    main()
