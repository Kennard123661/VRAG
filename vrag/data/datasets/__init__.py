
def sort_videos_by_similarity(video_similarity: dict):
    sorted_videos = [db_id for db_id, _ in sorted(video_similarity.items(), key=lambda item: item[1], reverse=True)]
    return sorted_videos
