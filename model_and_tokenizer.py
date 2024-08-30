import os
os.environ["TRANSFORMERS_CACHE"] = "/cs/snapless/roys/lab_resources"
os.environ["HF_HOME"] = "/cs/snapless/roys/lab_resources"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from config import config



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def get_tokenizer_for_ft():
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["pretrained_models_path"], token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_fine_tuned_model_for_inference():
    model = AutoModelForCausalLM.from_pretrained(config["finetuned_models_path"]).to(device)
    return model


def get_fine_tuned_tokenizer_for_inference():
    tokenizer = AutoTokenizer.from_pretrained(config["finetuned_models_path"])
    return tokenizer


def get_raw_model_for_zero_shot_inference():
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], cache_dir=config["pretrained_models_path"], token=os.getenv("HF_TOKEN")).to(device)
    return model


def get_raw_tokenizer_for_zero_shot_inference():
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["pretrained_models_path"], token=os.getenv("HF_TOKEN"))
    return tokenizer
