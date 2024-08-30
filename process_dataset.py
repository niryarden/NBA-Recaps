from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from config import config
from model_and_tokenizer import get_tokenizer_for_ft


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
        play_by_plays = examples["input"]
        inputs = [prompt_structure.format(play_by_play=play_by_play) for play_by_play in play_by_plays]
        outputs = examples["output"]
        model_inputs = tokenizer(inputs, max_length=config["max_input_length"], truncation=True, padding=True)
        labels = tokenizer(outputs, max_length=config["max_output_length"], truncation=True, padding=True).input_ids
        # Replace padding token id's of the labels by -100 so it's ignored by the loss function
        labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label] for label in labels]
        model_inputs["labels"] = labels
        return model_inputs
    
    raw_datasets = load_dataset(config["dataset_name"], cache_dir=config["dataset_path"], trust_remote_code=True)
    raw_datasets = raw_datasets.remove_columns("metadata")
    del raw_datasets["unsupervised"]
    datasets = raw_datasets.map(preprocess_function, batched=True)
    datasets.save_to_disk(config["processed_dataset_path"])


if __name__ == "__main__":
    process_dataset(get_tokenizer_for_ft())
