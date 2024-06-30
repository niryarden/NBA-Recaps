import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig)
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
import evaluate


model_name = "google/gemma-2b"
dataset_name = "nir-yar/nba-pbp-to-recap"
context_length = 8192
model_path = "/cs/labs/roys/nir.yarden/cache"
dataset_path = "/cs/labs/roys/nir.yarden/cache"
weigths_path = "/cs/labs/roys/nir.yarden/anlp-project/model_weights/"
output_dir = "/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/output_dir"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
# metric = evaluate.load("perplexity")

# hyperparameters
max_length = context_length // 16
lora_alpha = 32
lora_r = 64
lora_dropout = 0.05
learning_rate = 2e-5
weight_decay = 0.01
batch_size = 4
num_train_epochs = 3


# other potential models:
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-7b-hf"


def get_model():
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path).to(device)
    lora_config = LoraConfig(
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    return model

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    tokenizer.pad_token = tokenizer.eos_token\
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))
    return tokenizer


def get_datasets(tokenizer):
    def preprocess_function(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        labels = tokenizer(outputs, max_length=max_length, truncation=True, padding="max_length").input_ids
        # Replace padding token id's of the labels by -100 so it's ignored by the loss function
        labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label] for label in labels]
        model_inputs["labels"] = labels
        return model_inputs
    
    raw_datasets = load_dataset(dataset_name, cache_dir=dataset_path, trust_remote_code=True)
    raw_datasets = raw_datasets.remove_columns("metadata")
    del raw_datasets["unsupervised"]
    datasets = raw_datasets.map(preprocess_function,batched=True)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    # todo: temporary (smaller datasets)
    train_dataset = train_dataset.select(range(100))
    eval_dataset = eval_dataset.select(range(10))
    test_dataset = test_dataset.select(range(10))

    return train_dataset, eval_dataset, test_dataset


# def compute_metrics(p):
#     preds = np.argmax(p.predictions, axis=1)
#     return metric.compute(predictions=preds, references=p.label_ids)


def get_trainer():
    model = get_model()
    tokenizer = get_tokenizer()
    train_dataset, eval_dataset, test_dataset = get_datasets(tokenizer)
    training_args = TrainingArguments(
        output_dir="./output_dir",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )
    return trainer, model, test_dataset
    


def main():
    trainer, model, test_dataset = get_trainer()
    train_result = trainer.train()
    # trainer.save_model()
    print(train_result)
    train_metrics = train_result.metrics
    print(train_metrics)
    eval_metrics = trainer.evaluate()
    print(eval_metrics)


if __name__ == "__main__":
    main()
