import argparse
from transformers import AutoModel, AutoTokenizer
import os

def save_model_to_disk(model_name, save_path):
    # Load the model and tokenizer from Hugging Face
    print(f"Loading model and tokenizer for {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a subfolder with the model name
    model_save_path = os.path.join(save_path, model_name)

    # Save the model and tokenizer to the specified path
    print(f"Saving model and tokenizer to {model_save_path}...")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved successfully to {model_save_path}.")

def main():
    parser = argparse.ArgumentParser(description="Load a model from Hugging Face and save it to disk.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the model and tokenizer.")

    args = parser.parse_args()

    save_model_to_disk(args.model_name, args.save_path)

if __name__ == "__main__":
    main()