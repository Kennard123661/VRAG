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

SIMILARITY_MEASURES = ['chamfer', 'symmetric-chamfer']


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True)
    argparser.add_argument('--dataset', type=str, choices=['ccweb', 'evve', 'fivr5k', 'fivr200k'], required=True)
    argparser.add_argument('--threshold', type=float, default=0.75)
    args = argparser.parse_args()
    return args


def get_embedding_similarities(all_query_shot_embeddings, db_shot_embeddings: np.ndarray, query_num_shots: list):
    """
    Args:
        all_query_shot_embeddings: [S1 + ... + Sq] x D shot embeddings for all query videos 1 to q
        db_shot_embeddings: Sd x D shot embedding for one database video
        query_num_shots: [S1, ..., Sq] list of the number of shots in each query video
    Returns:
        list of q similarities between the each query video and the db video. The similarity is taken as the maximum
        shot similarity between the videos.
    """
    assert len(all_query_shot_embeddings.shape) == len(db_shot_embeddings.shape) == 2
    num_q_shots = all_query_shot_embeddings.shape[0]
    num_db_shots = db_shot_embeddings.shape[0]
    shot_similarities = cosine_similarity(all_query_shot_embeddings, db_shot_embeddings)\
        .reshape(num_q_shots, num_db_shots)
    num_queries = len(query_num_shots)
    chamfer_similarities = np.zeros(num_queries, dtype=float)
    sym_chamfer_similarities = np.zeros(num_queries, dtype=float)

    start = 0
    for i, num_shots in enumerate(query_num_shots):
        end = start + num_shots
        query_db_similarities = shot_similarities[start:end, :]
        start = end

        # take the maximum shot similarity score as the similarities between two shots
        chamfer_similarities[i] = np.mean(np.max(query_db_similarities, axis=1)).item()
        sym_chamfer_similarities[i] = (chamfer_similarities[i] + np.mean(np.max(query_db_similarities, axis=0)).item()) / 2

    similarities = {
        'chamfer': chamfer_similarities,
        'symmetric-chamfer': sym_chamfer_similarities
    }
    return similarities


def create_embeddings(dataset: str, config: str, threshold: float):
    """
    Creates video embeddings from R-MAC features

    Args:
        dataset: dataset to generate video embeddings
        config: configuration file containing hyperparameters
        threshold: similarity threshold for shot boundary algorithm

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
    postfix = str(int(threshold * 100))
    result_dir = os.path.join(RESULT_DIR, config, dataset)
    embedding_dir = os.path.join(result_dir, 'shot-embeddings-{}'.format(postfix))
    query_shot_embeddings = [np.load(os.path.join(embedding_dir, '{}.npy'.format(vid)))
                             for vid in query_video_ids]
    query_num_shots = [shot_embeddings.shape[0] for shot_embeddings in query_shot_embeddings]
    all_query_shot_embeddings = np.concatenate(query_shot_embeddings, axis=0)

    similarity_dict = {}
    for measure in SIMILARITY_MEASURES:
        similarity_dict[measure] = {query_id: {} for query_id in query_video_ids}

    print('INFO: computing similarities for {} shot embeddings...'.format(dataset))
    similarity_file = os.path.join(result_dir, 'shot-similarities-{}.json'.format(postfix))
    for db_id in tqdm(db_video_ids):
        db_shot_embeddings = np.load(os.path.join(embedding_dir, db_id + '.npy'))
        similarities = get_embedding_similarities(all_query_shot_embeddings=all_query_shot_embeddings,
                                                  db_shot_embeddings=db_shot_embeddings,
                                                  query_num_shots=query_num_shots)
        for measure in SIMILARITY_MEASURES:
            for i, query_id in enumerate(query_video_ids):
                similarity_dict[measure][query_id][db_id] = float(similarities[measure][i])

    with open(similarity_file, 'w') as f:
        json.dump(similarity_dict, f)

    print('INFO: evaluating similarities for {} Shots!'.format(dataset))
    for measure in SIMILARITY_MEASURES:
        print('INFO: evaluating shot {} similarity:'.format(measure))
        video_similarities = similarity_dict[measure]
        dset.evaluate(video_similarities=video_similarities)


def main():
    args = _parse_arguments()
    create_embeddings(dataset=args.dataset, config=args.config, threshold=args.threshold)


if __name__ == '__main__':
    main()
