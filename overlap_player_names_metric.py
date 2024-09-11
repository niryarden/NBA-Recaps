from utils import load_nba_dataset

from config import config


class OverLapPlayerNamesMetric:
    def __init__(self, dataset_name=config["dataset_name"]):
        self.dataset, _, _, _ = load_nba_dataset(dataset_name)
        self.game_scores = {}

    @staticmethod
    def extract_names_from_text(names_list, text):
        # Normalize the text to make matching case-insensitive (optional)
        text_lower = text.lower()

        # Find all names from the list that appear in the text
        extracted_names = [name for name in names_list if name.lower() in text_lower]

        return extracted_names

    def compute_metric(self, reference_text, generated_text, names_list, game_id=None):
        reference_names = set(self.extract_names_from_text(names_list, reference_text))
        generated_names = set(self.extract_names_from_text(names_list, generated_text))

        overlap = reference_names.intersection(generated_names)
        score = len(overlap) / len(reference_names) if reference_names else 0
        if game_id:
            self.game_scores[game_id] = score

        return score

    def get_top_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get, reverse=True)[:5]

    def get_lowest_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get)[:5]

    def get_average_score(self):
        return sum(self.game_scores.values()) / len(self.game_scores)
