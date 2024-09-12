import re
import json
import nltk
import string
from config import config
from nltk.stem import WordNetLemmatizer
from collections import Counter
from utils import load_nba_dataset, DEFAULT_VALUE
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

nltk_stopwords = set(stopwords.words('english'))
N_WORDS_TO_EXTRACT = 500
OUTPUT_KEY = 'output'
CUSTOM_STOPWORDS = {'said', 'get', 'good', 'nba', 'make', 'really', 'think', 'also', 'like', 'u', 'guy', 'in', 'year',
                    'go', 'went', 'great', 'without', 'best', 'better', 'thing', 'came', 'took'}
STOP_WORDS = nltk_stopwords.union(CUSTOM_STOPWORDS)


class OverLapWordsMetric:
    def __init__(self, dataset=config["dataset_name"]):
        self.lemmatizer = WordNetLemmatizer()
        self.table = str.maketrans('', '', string.punctuation)
        self.dataset, _, _, _ = load_nba_dataset(dataset)
        self.word_counter = self._count_words_from_dataset()
        self.nba_words = self.word_counter.keys()
        self.game_scores = {}

    def normalize_text(self, text):
        text = re.sub(r'\b\d+\b', '', text)  # Remove digits
        text = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', '', text,
                      flags=re.IGNORECASE)  # Remove written numbers

        words = re.findall(r'\b\w+\b', text.lower())  # Extract words and convert to lowercase
        normalized_words = [self.lemmatizer.lemmatize(word.translate(self.table)) for word in words if
                            self.lemmatizer.lemmatize(word.translate(self.table)) not in STOP_WORDS]
        unnormalized_words = [word.translate(self.table) for word in words if
                            word.translate(self.table) not in STOP_WORDS]
        return normalized_words + unnormalized_words

    def extract_and_normalize_patterns(self, text, pattern):
        matches = re.findall(pattern, text)
        normalized_matches = [self.lemmatizer.lemmatize(match) for match in matches]  # Lemmatize the matches
        return normalized_matches

    def _count_words_from_json(self, file_path, n_most_common=N_WORDS_TO_EXTRACT):
        word_counter = Counter()

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get(OUTPUT_KEY, '')
                normalized_words = self.normalize_text(text)
                word_counter.update(normalized_words)

        if n_most_common:
            return dict(word_counter.most_common(n_most_common))

        return word_counter

    def _count_words_from_dataset(self, n_most_common=N_WORDS_TO_EXTRACT):
        word_counter = Counter()

        for line in self.dataset:
            text = line.get(OUTPUT_KEY, '')
            normalized_words = self.normalize_text(text)
            word_counter.update(normalized_words)

        if n_most_common:
            word_counter = dict(word_counter.most_common(n_most_common))

        print(word_counter.keys())
        return word_counter

    def compute_metric(self, reference_text, generated_text, game_id=None):
        reference_words = set(self.normalize_text(reference_text))
        generated_words = set(self.normalize_text(generated_text))

        nba_words_in_reference = reference_words.intersection(self.nba_words)
        nba_words_in_generated = generated_words.intersection(self.nba_words)

        overlap = nba_words_in_reference.intersection(nba_words_in_generated)
        score = len(overlap) / len(nba_words_in_reference) if nba_words_in_reference else DEFAULT_VALUE
        if game_id:
            self.game_scores[game_id] = score

        return score

    def get_top_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get, reverse=True)[:5]

    def get_lowest_5_generated_recaps(self):
        return sorted(self.game_scores, key=self.game_scores.get)[:5]

    def get_average_score(self):
        return sum(self.game_scores.values()) / len(self.game_scores)
