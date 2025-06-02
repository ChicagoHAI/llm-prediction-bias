#!/usr/bin/env python3
"""
Extract and average representations from a specified layer at the last token.
Uses PyVene's CollectIntervention to gather activations.
"""

import sys
sys.path.append("../pyvene")

import argparse
import os
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import pyvene as pv
import pandas as pd
import re
from utils import NAMES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and average representations from a model layer"
    )
    
    # Model and layer specification
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Name or path of the model (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--layer", 
        type=int, 
        required=True,
        help="Layer number to extract representations from (0-indexed)"
    )
    
    # Dataset specification
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to CSV dataset file"
    )
    
    # Processing options
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=-1,
        help="Maximum number of samples to process (-1 for all)"
    )
    parser.add_argument(
        "--race", 
        type=str, 
        default=None,
        help="Specific race to extract representations for (e.g., 'white', 'black'). If None, uses all samples."
    )
    
    # Output specification
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Path to save the averaged representation (.pt or .npy)"
    )
    
    # Device and optimization
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use ('cuda', 'cpu', or 'auto')"
    )
    
    return parser.parse_args()


def extract_race_from_name(text: str) -> str:
    """Extract race from name mentioned in the text using utils.py mapping."""
    # Create flattened name-to-race mapping from NAMES dictionary
    name_to_race = {}
    for race, names in NAMES.items():
        for name in names:
            name_to_race[name.lower()] = race.lower()

    # Extract names from text using regex (looking for names after "admit")
    name_pattern = r'admit\s+(\w+)'
    match = re.search(name_pattern, text, re.IGNORECASE)
    
    if match:
        name = match.group(1).lower()
        return name_to_race.get(name, 'unknown')
    
    return 'unknown'


def load_data(dataset_path: str, text_column: str, 
              max_samples: int = -1, race: str = None) -> Dataset:
    """Load dataset from CSV file, optionally filtering by race extracted from names."""
    if not dataset_path.endswith('.csv'):
        raise ValueError("Only CSV files are supported")
    
    df = pd.read_csv(dataset_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Extract race from names in the text
    print("Extracting race information from names...")
    df['extracted_race'] = df[text_column].apply(extract_race_from_name)
    
    # Print race distribution
    race_counts = df['extracted_race'].value_counts()
    print(f"Race distribution: {race_counts.to_dict()}")
    
    # Filter by race if specified
    if race is not None:
        initial_count = len(df)
        df = df[df['extracted_race'] == race]
        print(f"Filtered from {initial_count} to {len(df)} samples for race: {race}")
        
        if len(df) == 0:
            raise ValueError(f"No samples found for race: {race}")
    
    dataset = Dataset.from_pandas(df)
    
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    
    return dataset


def setup_device(device_arg: str) -> torch.device:
    """Setup the computation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def extract_representations(
    model, tokenizer, dataset: Dataset, layer: int, 
    component: str, text_column: str, batch_size: int, 
    device: torch.device
) -> torch.Tensor:
    """Extract representations from the specified layer at last token."""
    
    # Setup PyVene intervention for collecting activations
    config_collect = pv.IntervenableConfig([{
        "layer": layer,
        "component": component,
        "intervention_type": pv.CollectIntervention
    }])
    
    intervenable = pv.IntervenableModel(config_collect, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    
    # Setup data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_representations = []
    
    print(f"Extracting representations from layer {layer}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Tokenize the batch
            texts = batch[text_column]
            tokens = tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True,
            ).to(device)
            
            # Get sequence length for last token position
            seq_len = tokens.input_ids.shape[1]
            last_token_pos = seq_len - 1
            
            # Extract representations using PyVene
            intervenable_out = intervenable(
                tokens,
                unit_locations={"base": last_token_pos}
            )
            
            # intervenable_out is ((base_outputs, collected_activations), None)
            collected_activations = intervenable_out[0][1]
            
            # collected_activations is a list, concatenate to get tensor
            batch_representations = torch.concatenate(collected_activations)
            
            all_representations.append(batch_representations.cpu())
    
    # Concatenate all representations
    all_representations = torch.cat(all_representations, dim=0)
    
    print(f"Extracted {all_representations.shape[0]} representations "
          f"of dimension {all_representations.shape[1]}")
    
    return all_representations


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    if device.type != "cuda" or "device_map" not in locals():
        model = model.to(device)
    model.eval()
    
    # Validate layer number
    if args.layer >= config.num_hidden_layers:
        raise ValueError(
            f"Layer {args.layer} exceeds model depth "
            f"({config.num_hidden_layers} layers)"
        )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_data(
        args.dataset_path, 
        "base", 
        args.max_samples, args.race
    )
    print(f"Loaded {len(dataset)} samples")

    # Extract representations
    representations = extract_representations(
        model, tokenizer, dataset, args.layer, 
        "block_output", "base", 
        args.batch_size, device
    )
    
    # Calculate average representation
    avg_representation = representations.mean(dim=0)
    
    print(f"Average representation shape: {avg_representation.shape}")
    
    # Create save directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Modify save path to include race info if specified
    save_path = args.save_path
    if args.race is not None:
        # Insert race into filename before extension
        path_parts = args.save_path.rsplit('.', 1)
        if len(path_parts) == 2:
            save_path = f"{path_parts[0]}_{args.race}.{path_parts[1]}"
        else:
            save_path = f"{args.save_path}_{args.race}"
    
    # Save average representation
    torch.save(avg_representation, save_path)
    print(f"Saved average representation to: {save_path}")

if __name__ == "__main__":
    main() 