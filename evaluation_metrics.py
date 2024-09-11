from overlap_player_names_metric import OverLapPlayerNamesMetric
from overlap_words_metric import OverLapWordsMetric
from sbert_similarity_metric import SBERTSimilarityMetric

PLAYER_NAMES_KEY = "player_names"
SOURCE_TEXT_KEY = "source_text"
GAME_ID_KEY = "game_id"

class NBAMetric:
    def __init__(self):
        self.overlap_nba_words_metric = OverLapWordsMetric()
        self.overlap_player_names_metric = OverLapPlayerNamesMetric()
        self.sbert_metric = SBERTSimilarityMetric()

    def compute(self, references, predictions):
        for ref, pred in zip(references, predictions):
            game_id = ref[GAME_ID_KEY]
            player_names = ref[PLAYER_NAMES_KEY]
            ref_val = ref[SOURCE_TEXT_KEY]

            self.overlap_nba_words_metric.compute_metric(ref_val, pred, game_id=game_id)
            self.overlap_player_names_metric.compute_metric(ref_val, pred, player_names, game_id=game_id)
            self.sbert_metric.compute_metric(ref_val, pred, game_id=game_id)

        return {
            "name_overlap_avg_score": self.overlap_player_names_metric.get_average_score(),
            "nba_overlap_avg_score": self.overlap_nba_words_metric.get_average_score(),
            "sbert_similarity_avg_score": self.sbert_metric.get_average_score(),
        }

def load_results_dummy():
    references = []
    predictions = []

    for i in range(100):
        curr_ref = open(f"reference/reference_{i}").readlines()
        references.append(curr_ref)
        curr_pred = open(f"reference/reference_{i}").readlines()
        predictions.append(curr_pred)

    return predictions, references


if __name__ == "__main__":
    predictions, references = load_results_dummy()
    metric = NBAMetric()
    scores = metric.compute(references=references, predictions=predictions)
    print(scores)
