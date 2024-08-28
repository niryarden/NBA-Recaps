import json
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

N_WORDS_TO_EXTRACT = 100
OUTPUT_KEY = 'output'
STOP_WORDS = set(stopwords.words('english'))

def extract_words_from_jsonl(file_path):
    word_counter = Counter()
    table = str.maketrans('', '', string.punctuation)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get(OUTPUT_KEY, '')
            words = [word.lower().translate(table) for word in text.split() if len(word.lower().translate(table)) >= 3 and word.lower().translate(table) not in STOP_WORDS]
            word_counter.update(words)

    return word_counter


def get_most_common_words(file_path, n=N_WORDS_TO_EXTRACT):
    word_counter = extract_words_from_jsonl(file_path)
    return dict(word_counter.most_common(n))


file_path = '2020/2020_samples.jsonl'
most_common_words = get_most_common_words(file_path)
print(most_common_words.keys())
