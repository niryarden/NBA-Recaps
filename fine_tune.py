import torch
from transformers import Trainer, TrainingArguments
from config import config
from process_dataset import get_datasets
from model_and_tokenizer import get_model_for_ft, get_tokenizer_for_ft


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# metric = evaluate.load("perplexity")

# def compute_metrics(p):
#     preds = np.argmax(p.predictions, axis=1)
#     return metric.compute(predictions=preds, references=p.label_ids)


def get_trainer():
    model = get_model_for_ft()
    tokenizer = get_tokenizer_for_ft()
    train_dataset, eval_dataset, _ = get_datasets()
    training_args = TrainingArguments(
        output_dir=config["finetuned_models_path"],
        save_strategy="epoch",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_train_epochs"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
        # compute_metrics=compute_metrics,
    )
    return trainer, tokenizer


def main():
    trainer, tokenizer = get_trainer()
    trainer.train()
    trainer.save_model(config["finetuned_models_path"])
    tokenizer.save_pretrained(config["finetuned_models_path"])


if __name__ == "__main__":
    main()
