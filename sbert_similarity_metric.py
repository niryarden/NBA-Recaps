from sentence_transformers import SentenceTransformer, util
from utils import load_nba_dataset
from config import config


class SBERTSimilarityMetric:
    def __init__(self, dataset_name=config["dataset_name"], model_name='paraphrase-MiniLM-L6-v2'):
        self.dataset, _, _, _ = load_nba_dataset(dataset_name)
        self.model = SentenceTransformer(model_name)
        self.game_scores = {}

    def compute_metric(self, reference_text, generated_text, game_id=None):
        reference_embedding = self.model.encode(reference_text, convert_to_tensor=True)
        generated_embedding = self.model.encode(generated_text, convert_to_tensor=True)

        similarity = util.pytorch_cos_sim(reference_embedding, generated_embedding).item()

        if game_id:
            self.game_scores[game_id] = similarity

        return similarity

    def get_top_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get, reverse=True)[:5]

    def get_lowest_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get)[:5]

    def get_average_score(self):
        return sum(self.game_scores.values()) / len(self.game_scores) if self.game_scores else 0
