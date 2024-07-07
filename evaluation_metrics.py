from datasets import MetricInfo, Value
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


def extract_names_ner(text):
    # TODO: compare performance against simple regex function (this function is currently unused)
    ner_results = ner_pipeline(text)
    names = [result['word'] for result in ner_results if result['entity'].startswith('B-PER')]
    return set(names)


def extract_names(text):
    return re.findall(r'\b[A-Z][a-z]*\b', text)


nba_related_words = [
    'coach',
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
    return util.pytorch_cos_sim(ref_embedding, gen_embedding).item()


class NBAMetric:
    def __init__(self):
        # TODO: check other text similarity options
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def _info(self):
        return MetricInfo(
            description="Custom metric for NBA recap generation",
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

        return {
            "name_overlap_score": name_overlap_score,
            "nba_overlap_score": nba_overlap_score,
            "sbert_similarity_score": sbert_similarity_score
        }


def nba_metric():
    return NBAMetric()


# TODO: add more heuristics. for example, maybe use QA (such as "who scored most points?" etc)

if __name__ == "__main__":
    references = ["LeBron James scored 30 points in the NBA game last night."]
    predictions = ["LeBron James made 30 points in the basketball match yesterday."]

    metric = nba_metric()
    scores = metric._compute(references=references, predictions=predictions)
    print(scores)
