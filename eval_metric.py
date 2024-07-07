from datasets import MetricInfo, Value
import re
from sentence_transformers import SentenceTransformer, util

def extract_names(text):
    return re.findall(r'\b[A-Z][a-z]*\b', text)


nba_related_words = [
    'basketball', 'NBA', 'player', 'coach', 'team', 'game', 'season', 'score',
    'points', 'rebounds', 'assists', 'block', 'steal', 'dunk', 'three-pointer',
    'free throw', 'foul', 'court', 'hoop', 'dribble', 'shoot', 'pass', 'defense'
]


def compute_nba_words_overlap(reference_text, generated_text):
    reference_words = set(re.findall(r'\b\w+\b', reference_text.lower()))
    generated_words = set(re.findall(r'\b\w+\b', generated_text.lower()))

    nba_words_in_reference = reference_words.intersection(nba_related_words)
    nba_words_in_generated = generated_words.intersection(nba_related_words)

    overlap = nba_words_in_reference.intersection(nba_words_in_generated)
    return len(overlap) / len(nba_words_in_reference) if nba_words_in_reference else 0

def compute_sbert_similarity(reference_text, generated_text, model):
    ref_embedding = model.encode(reference_text, convert_to_tensor=True)
    gen_embedding = model.encode(generated_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(ref_embedding, gen_embedding).item()

    # Normalize cosine similarity to [0, 1]
    normalized_similarity = (similarity + 1) / 2
    return normalized_similarity

class NBAMetric:
    def __init__(self, name_weight, nba_weight, sbert_weight):
        assert (name_weight+nba_weight+sbert_weight == 1), "Weights should sum up to 1"
        self.name_weight = name_weight
        self.nba_weight = nba_weight
        self.sbert_weight = sbert_weight
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def _info(self):
        return MetricInfo(
            description="Custom metric for text generation: name extraction and NBA word overlap",
            citation="",
            inputs_description="Inputs should be a list of reference texts and generated texts.",
            features={
                "references": Value("string"),
                "predictions": Value("string"),
            }
        )

    def _compute(self, references, predictions):
        total_names = 0
        matched_names = 0
        total_nba_overlap = 0
        total_sbert_similarity = 0

        for ref, pred in zip(references, predictions):
            ref_names = set(extract_names(ref))
            pred_names = set(extract_names(pred))
            matched_names += len(ref_names.intersection(pred_names))
            total_names += len(ref_names)

            total_nba_overlap += compute_nba_words_overlap(ref, pred)
            total_sbert_similarity += compute_sbert_similarity(ref, pred, self.model)

        name_overlap_score = matched_names / total_names if total_names > 0 else 0
        nba_overlap_score = total_nba_overlap / len(references)
        sbert_similarity_score = total_sbert_similarity / len(references)

        combined_score = (self.name_weight * name_overlap_score) + (self.nba_weight * nba_overlap_score) + (
                    self.sbert_weight * sbert_similarity_score)

        return combined_score


def nba_metric(name_weight=0.2, nba_weight=0.2, sbert_weight=0.6):
    return NBAMetric(name_weight, nba_weight, sbert_weight)


if __name__ == "__main__":
    references = ["LeBron James scored 30 points in the NBA game last night."]
    predictions = ["LeBron James made 30 points in the basketball match yesterday."]

    metric = nba_metric()
    combined_score = metric._compute(references=references, predictions=predictions)
    print(f"Combined Score: {combined_score}")
