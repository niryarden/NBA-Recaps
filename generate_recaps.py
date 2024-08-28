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
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    model, tokenizer = get_model_and_tokenizer(finetuned=False)
    _, _, test_dataset = get_datasets()

    sample = test_dataset[0]
    generated_output = generate_output(model, tokenizer, sample)

    print("Reference Output:")
    print(sample["output"])
    print("\nGenerated Output:")
    print(generated_output)


if __name__ == "__main__":
    main()
