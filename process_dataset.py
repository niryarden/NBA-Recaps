import os
os.environ["TRANSFORMERS_CACHE"] = "/cs/snapless/roys/lab_resources"
os.environ["HF_HOME"] = "/cs/snapless/roys/lab_resources"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from config import config
from model_and_tokenizer import get_tokenizer_for_ft
from preprocess.preprocessing import Preprocessor

prompt_structure = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 1 September 2024
You are a journalist that specializes with the NBA.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please write an NBA game recap based on the following play-by-play data.
**Play-by-play:**
{play_by_play}
**End of play-by-play:**
**Game Recap:**
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def get_datasets():
    datasets = load_from_disk(config["processed_dataset_path"])
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    return train_dataset, eval_dataset, test_dataset


def process_dataset(tokenizer):
    max_input_length = config["max_input_length"]
    prompt_tokens = tokenizer.encode(prompt_structure, add_special_tokens=True)
    available_length = max_input_length - len(prompt_tokens)

    def preprocess_function(examples):
        preprocessor = Preprocessor()
        play_by_plays = examples["input"]
        outputs = examples["output"]
        processed_inputs = []
        for play_by_play in play_by_plays:
            processed_play_by_play = preprocessor.preprocess(play_by_play)
            play_by_play_tokens = tokenizer.encode(processed_play_by_play, add_special_tokens=False)
            truncated_tokens = play_by_play_tokens[:available_length]
            truncated_play_by_play = tokenizer.decode(truncated_tokens)
            full_prompt = prompt_structure.format(play_by_play=truncated_play_by_play)
            processed_inputs.append(full_prompt)

        model_inputs = tokenizer(processed_inputs, max_length=max_input_length, truncation=True, padding=True)
        labels = tokenizer(outputs, max_length=config["max_output_length"], truncation=True, padding=True).input_ids
        labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label] for label in labels]
        model_inputs["labels"] = labels
        model_inputs["metadata"] = examples["metadata"]
        model_inputs["output"] = examples["output"]
        return model_inputs
    
    raw_datasets = load_dataset(config["dataset_name"], cache_dir=config["dataset_path"], trust_remote_code=True)
    del raw_datasets["unsupervised"]
    datasets = raw_datasets.map(preprocess_function, batched=True)
    datasets.save_to_disk(config["processed_dataset_path"])


if __name__ == "__main__":
    process_dataset(get_tokenizer_for_ft())
