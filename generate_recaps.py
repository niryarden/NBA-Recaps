import os
os.environ["TRANSFORMERS_CACHE"] = "/cs/snapless/roys/lab_resources"
os.environ["HF_HOME"] = "/cs/snapless/roys/lab_resources"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import json
import torch
from config import config
from process_dataset import get_datasets
from model_and_tokenizer import get_fine_tuned_model_for_inference, get_fine_tuned_tokenizer_for_inference, get_raw_model_for_zero_shot_inference, get_raw_tokenizer_for_zero_shot_inference


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_and_tokenizer(finetuned=False):
    if finetuned:
        return get_fine_tuned_model_for_inference(), get_fine_tuned_tokenizer_for_inference()
    return get_raw_model_for_zero_shot_inference(), get_raw_tokenizer_for_zero_shot_inference()


def generate_output(model, tokenizer, sample):
    outputs = model.generate(
        torch.tensor(sample["input_ids"]).unsqueeze(0).to(device),
        attention_mask=torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device),
        max_new_tokens=config["max_length"],
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    recap = outputs[0][len(sample["input_ids"]):]
    generated_text = tokenizer.decode(recap, skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_index", type=int, default=0)
    parser.add_argument("--last_index", type=int, default=567)
    args =  parser.parse_args()
    first_index, last_index = args.first_index, args.last_index
    print(args)

    model, tokenizer = get_model_and_tokenizer(finetuned=False)
    _, _, test_dataset = get_datasets()

    for i in range(first_index, last_index + 1):
        if i >= len(test_dataset):
            break
        print(f"sample no. {i}")
        sample = test_dataset[i]
        to_save = {}
        to_save["metadata"] = sample["metadata"]
        to_save["reference_recap"] = sample["output"]
        to_save["generated_recap"] = generate_output(model, tokenizer, sample)
        as_json = json.dumps(to_save)
        with open(f"/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/recaps/test_sample_{i}.json", 'w') as f:
            f.write(as_json)


if __name__ == "__main__":
    main()
