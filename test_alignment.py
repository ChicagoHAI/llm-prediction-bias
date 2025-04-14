import argparse
import os
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import pyvene as pv

from utils import load_alignment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", help="""Path the the directory containing
                        the dataset files.""")
    parser.add_argument("--model_name", type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')
    parser.add_argument("--alignment_path")
    parser.add_argument("--intervention_type", choices=["bdas", "das"], default="bdas")
    parser.add_argument("--interchange_dim", type=int)

    parser.add_argument("--horizontal_start", type=int, default=0)
    parser.add_argument("--horizontal_end", type=int, default=50)
    parser.add_argument("--horizontal_step", type=int, default=1)

    parser.add_argument("--vertical_start", type=int, default=0)
    parser.add_argument("--vertical_end", type=int, default=-1)
    parser.add_argument("--vertical_step", type=int, default=1)

    parser.add_argument("--extra_steps", 
                        help="""The number of steps before {h_pos} to search.""", 
                        default=4, type=int)
    parser.add_argument("--train_end", action='store_true', 
                        help="Whether to train on tokens near the prompt's end."
                        )

    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--batch_size", help="""Training batch size.""",
                        default=32, type=int)
    parser.add_argument("--results_save_path", 
                        help="""Path to the directory to save
                        the test accuracies.""")

    args = parser.parse_args()

    ds_path = args.dataset_path
    model_name = args.model_name
    align_path = args.alignment_path
    vene_type = args.intervention_type
    interchange_dim = args.interchange_dim

    h_start = args.horizontal_start
    h_end = args.horizontal_end
    h_step = args.horizontal_step
    num_extra_steps = args.extra_steps
    train_end = args.train_end

    v_start = args.vertical_start
    v_end = args.vertical_end
    v_step = args.vertical_step

    n_test = args.n_test
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    llama = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,  # save memory
    )
    _ = llama.to(device)
    _ = llama.eval()

    ds = load_dataset('csv', data_files={
        'test': os.path.join(ds_path, 'test.csv'),
    })

    if n_test > 0:
        indices = np.arange(n_test)
        ds['test'] = ds['test'].shuffle(seed=42).select(indices)

    test_loader = DataLoader(ds['test'], batch_size=batch_size)

    if v_end == -1:
        v_end = llama.config.num_hidden_layers

    extra_steps = num_extra_steps * h_step
    layers = list(range(v_start, v_end+1, v_step))
    positions = list(range(h_start-extra_steps, h_end+1, h_step))

    if train_end:
        positions += list(range(-1-extra_steps, 0, h_step))

    # we search over layers and token positions
    for layer in layers:
        for position in positions:
            args.save_name = f"layer_{layer}_token_{position}"

            print(args.save_name)

            align_path = os.path.join(args.alignment_path, args.save_name)

            if vene_type == "bdas":
                # Boundless rotated interchange intervention
                config_bdas = pv.IntervenableConfig(
                    [{
                        "layer": layer,
                        "component": "block_output",
                        "intervention_type": pv.BoundlessRotatedSpaceIntervention,
                    }]
                )
                vene_bdas = load_alignment(align_path, config_bdas, llama)
                vene_bdas.set_device(device)
                vene_bdas.disable_model_gradients()

                vene = vene_bdas
                print("Using boundless DAS!")

            elif vene_type == "das":
                # Vanilla DAS
                config_das = pv.IntervenableConfig(
                    [{
                        "layer": layer,
                        "component": 'block_output',
                        "intervention_type": pv.RotatedSpaceIntervention,
                    }]
                )
                vene_das = load_alignment(align_path, config_das, llama,
                                        alignment_type=pv.RotatedSpaceIntervention)
                vene_das.set_device(device)
                vene_das.disable_model_gradients()

                vene = vene_das
                print("Using vanilla DAS!")
                
            # eval
            with torch.no_grad():
                iterator = tqdm(test_loader, desc="Evaluating")
                all_preds = []
                all_labels = []
                
                for example in iterator:
                    base_tokens = tokenizer(example['base'], 
                                            return_tensors='pt', 
                                            padding=True).to(device)
                    source_tokens = tokenizer(example['source'], 
                                            return_tensors='pt', 
                                            padding=True).to(device)
                    
                    base = base_tokens.input_ids
                    src = source_tokens.input_ids

                    if position < 0:
                        base_pos = base.shape[1] + position
                        src_pos = src.shape[1] + position
                    else:
                        base_pos = src_pos = position
                
                    _, counterfactual_outputs = vene(
                        base_tokens,
                        [source_tokens],
                        unit_locations={"sources->base": (src_pos, base_pos)},
                    )

                    logits = counterfactual_outputs.logits[:, -1]
                    preds = logits.argmax(dim=-1).detach().cpu().numpy()

                    all_preds.append(preds)
                    all_labels.append(example['src_label'])
                    
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                acc = accuracy_score(all_preds, all_labels)

            os.makedirs(args.results_save_path, exist_ok=True)
            with open(os.path.join(args.results_save_path, args.save_name + ".txt"), 'w') as fw:
                fw.write(f"Final test accuracy: {acc:.2f}")