import json

from overlap_player_names_metric import OverLapPlayerNamesMetric
from overlap_words_metric import OverLapWordsMetric
from sbert_similarity_metric import SBERTSimilarityMetric
from utils import PredictedGame

PLAYER_NAMES_KEY = "player_names"
SOURCE_TEXT_KEY = "source_text"
GAME_ID_KEY = "game_id"


class NBAMetric:
    def __init__(self):
        self.overlap_nba_words_metric = OverLapWordsMetric()
        self.overlap_player_names_metric = OverLapPlayerNamesMetric()
        self.sbert_metric = SBERTSimilarityMetric()

    def compute(self, games):
        results = {}
        for game in games:
            score1 = self.overlap_nba_words_metric.compute_metric(game.reference, game.prediction, game_id=game.game_id)
            score2 = self.overlap_player_names_metric.compute_metric(game.reference, game.prediction, game.player_names,
                                                                     game_id=game.game_id)
            score3 = self.sbert_metric.compute_metric(game.reference, game.prediction, game_id=game.game_id)
            results[game.game_id] = [score1, score2, score3]

        return results


def load_results():
    games = []
    for i in range(10):
        with open(f"recaps/test_sample_{i}.json") as f:
            data = json.load(f)
            curr_metadata = data["metadata"]
            game_id = curr_metadata["game_id"]
            players_names = curr_metadata['home_team_players_names'] + curr_metadata['home_team_names_I'] + \
                            curr_metadata['away_team_players_names'] + curr_metadata['away_team_names_I']
            curr_game = PredictedGame(game_id, data["reference_recap"], data["generated_recap"], players_names)

            games.append(curr_game)

    return games


if __name__ == "__main__":
    games = load_results()
    metric = NBAMetric()
    scores = metric.compute(games)
    print(scores)
