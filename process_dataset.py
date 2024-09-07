from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from config import config
from model_and_tokenizer import get_tokenizer_for_ft
from preprocess.preprocessing import Preprocessor

prompt_structure = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 1 September 2024
You are a journalist that specializes with the NBA.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you please write an NBA game recap based on the following play-by-play data:
{play_by_play}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def get_datasets():
    datasets = load_from_disk(config["processed_dataset_path"])
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    # todo: temporary (smaller datasets)
    train_dataset = train_dataset.select(range(100))
    eval_dataset = eval_dataset.select(range(10))
    test_dataset = test_dataset.select(range(10))

    return train_dataset, eval_dataset, test_dataset


def process_dataset(tokenizer):
    def preprocess_function(examples):
        preprocessor = Preprocessor()
        play_by_plays = examples["input"]
        outputs = examples["output"]
        max_length = config["max_input_length"]

        # Calculate available space for play_by_play
        prompt_template = prompt_structure.replace("<placeholder>", "")
        prompt_tokens = tokenizer.encode(prompt_template, add_special_tokens=True)
        available_length = max_length - len(prompt_tokens)

        processed_inputs = []
        for play_by_play in play_by_plays:
            # Preprocess the play-by-play
            processed_play_by_play = preprocessor.preprocess(play_by_play)

            # Encode the processed play-by-play
            play_by_play_tokens = tokenizer.encode(processed_play_by_play, add_special_tokens=False)

            # Truncate the encoded play-by-play
            truncated_tokens = play_by_play_tokens[:available_length]

            # Decode the truncated play-by-play back to text
            truncated_play_by_play = tokenizer.decode(truncated_tokens)

            # Use string formatting to insert the truncated play-by-play into the full prompt
            full_prompt = prompt_structure.format(play_by_play=truncated_play_by_play)

            processed_inputs.append(full_prompt)

        # Tokenize the processed inputs
        model_inputs = tokenizer(processed_inputs, max_length=max_length, truncation=True, padding=True)

        # Process labels as before
        labels = tokenizer(outputs, max_length=config["max_output_length"], truncation=True, padding=True).input_ids
        labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label] for label in labels]
        model_inputs["labels"] = labels
    
    raw_datasets = load_dataset(config["dataset_name"], cache_dir=config["dataset_path"], trust_remote_code=True)
    raw_datasets = raw_datasets.remove_columns("metadata")
    del raw_datasets["unsupervised"]
    datasets = raw_datasets.map(preprocess_function, batched=True)
    datasets.save_to_disk(config["processed_dataset_path"])


if __name__ == "__main__":
    process_dataset(get_tokenizer_for_ft())
