from overlap_player_names_metric import OverLapPlayerNamesMetric
from overlap_words_metric import OverLapWordsMetric
from plot_utils import analyze_metrics
from qa_metric import QASimilarityMetric
from sbert_similarity_metric import SBERTSimilarityMetric
from utils import load_results_from_files


class NBAMetrics:
    def __init__(self):
        self.overlap_nba_words_metric = OverLapWordsMetric()
        self.overlap_player_names_metric = OverLapPlayerNamesMetric()
        self.sbert_metric = SBERTSimilarityMetric()
        self.qa_metric = QASimilarityMetric()

    def compute_metrics(self):
        games_df = load_results_from_files()

        # Compute the scores for each metric
        games_df['nba_overlap_score'] = games_df.apply(lambda row: self.overlap_nba_words_metric.compute_metric(
            row['reference_recap'], row['generated_recap'], game_id=row['game_id']), axis=1)

        games_df['player_names_overlap_score'] = games_df.apply(
            lambda row: self.overlap_player_names_metric.compute_metric(
                row['reference_recap'], row['generated_recap'], row['player_names'], game_id=row['game_id']), axis=1)

        games_df['sbert_similarity_score'] = games_df.apply(lambda row: self.sbert_metric.compute_metric(
            row['reference_recap'], row['generated_recap'], game_id=row['game_id']), axis=1)

        return games_df[['game_id', 'nba_overlap_score', 'player_names_overlap_score', 'sbert_similarity_score']]


if __name__ == "__main__":
    metric = NBAMetrics()
    results = metric.compute_metrics()
    analyze_metrics(results)
