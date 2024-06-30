config = {
    "model_name": "google/gemma-2b",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
    "context_length": 8192,
    "pretrained_models_path": "/cs/labs/roys/nir.yarden/cache",
    "finetuned_models_path": "/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/trained_models",
    "dataset_path": "/cs/labs/roys/nir.yarden/cache",
    "processed_dataset_path": "/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/processed_dataset",
    "dataset_name": "nir-yar/nba-pbp-to-recap",
    "max_length": 8192 - 1024,
    "lora_alpha": 32,
    "lora_r": 64,
    "lora_dropout": 0.05,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "batch_size": 4,
    "num_train_epochs": 3,
    "warmup_steps": 500,
    "lr_scheduler_type": "linear"
}

# other potential models:
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-7b-hf"
