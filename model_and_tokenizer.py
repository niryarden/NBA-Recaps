import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from config import config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lora_config = LoraConfig(
    target_modules=config["target_modules"],
    task_type=config["task_type"],
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
)


def get_model_for_ft():
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], cache_dir=config["pretrained_models_path"], token=os.getenv("HF_TOKEN")).to(device)
    lora_config = LoraConfig(
        target_modules=config["target_modules"],
        task_type=config["task_type"],
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )
    model = get_peft_model(model, lora_config)
    return model


def get_model_for_inference():
    return AutoModelForCausalLM.from_pretrained(config["finetuned_models_path"]).to(device)


def get_tokenizer_for_ft():
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["pretrained_models_path"], token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))
    return tokenizer


def get_tokenizer_for_inference():
    tokenizer = AutoTokenizer.from_pretrained(config["finetuned_models_path"])
    return tokenizer
