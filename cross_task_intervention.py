import os
import random
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
import einops

import sys

sys.path.append('../pyvene/')
import pyvene as pv

# from eval_alignment import load_alignment
from utils import get_bdas_params, load_alignment, get_race


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input arguments for cross-task intervention on HireDec.")

    parser.add_argument('--model_name', type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')
    parser.add_argument("--alignment_path", 
                        help="""Path to the directory 
                        containing the saved alignment.""")
    parser.add_argument("--src_alignment_path")
    parser.add_argument("--dataset_path", 
                        help="""Path to the directory containing
                        the counterfactual dataset files.""")
    parser.add_argument("--interchange_dim", type=int, default=2000)
    parser.add_argument("--base_task",
                        choices=['Admissions', 'HireDec', 'Hiring', 'HireDecNames', 'DiscrimEval', 'RaceQA', 'HiringRaceOffset'],
                        default="Admissions")

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--generate', action='store_true')

    # parser.add_argument('--source_reduction', type=str, 
    #                     choices=["selection", "mean", "zero-ablation"], 
    #                     default="selection")
    
    parser.add_argument('--method', 
                        choices=["bdas", "das", "full", 
                                 "probing", "random"
                                 ], default="bdas")
    parser.add_argument('--intervention', choices=['interchange', 'neuron-avg', 'zero-ablate', 'noising', 'var-avg', 'fix-race', 'prompting'], default='interchange')
    parser.add_argument('--prompt', type=str, default="")

    parser.add_argument('--collect_layer', type=int, default=2, 
                        help='Layer to collect activations from. This option only applies to DAS.')
    parser.add_argument('--collect_token', type=int, default=17, 
                        help='Position to collect activations from. This option only applies to DAS.')
    parser.add_argument("--use_right_padding", action='store_true')

    parser.add_argument('--patch_start', type=int, default=2, 
                        help='Start of patch layers range')
    parser.add_argument('--patch_end', type=int, default=4, 
                        help='End of patch layers range')
    
    parser.add_argument('--patch_tokens', type=str, 
                        choices=["naive", "precise", "random", "custom"], default="precise", 
                        help='Type of patch tokens')
    parser.add_argument('--patch_token_positions', nargs='+', type=int)
    
    # parser.add_argument('--concept_subspace', type=str, 
    #                     choices=["naive", "aligned", "custom"], default="aligned", 
    #                     help='Concept direction for intervention')

    parser.add_argument("--save_path", default="./", help="Path to the save directory")

    args = parser.parse_args()
    return args


args = parse_args()

bs = args.batch_size
n_test = args.n_test
# src_reduce = args.source_reduction
generate = args.generate

method = args.method
intervention = args.intervention
interchange_dim = args.interchange_dim
collect_layer = args.collect_layer
collect_token = args.collect_token
use_right_padding = args.use_right_padding # for when race position is varied
debias_prompt = args.prompt

patch_layers = range(args.patch_start, args.patch_end+1)
patch_tokens = args.patch_tokens
patch_token_positions = args.patch_token_positions
# concept_subspace = args.concept_subspace

ds_path = args.dataset_path
base_task = args.base_task
align_path = args.alignment_path
src_align_path = args.src_alignment_path
save_path = args.save_path

model_name = args.model_name
device = 'cuda:2'

# os.makedirs(save_path, exist_ok=True)

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

df = pd.read_csv(os.path.join(ds_path, 'test.csv'))
if n_test > 0:
    subset_size = n_test // 2
    df1 = df.loc[df['base_label'] == df['src_label']].sample(subset_size, replace=True)
    df2 = df.loc[df['base_label'] != df['src_label']].sample(subset_size, replace=True)
    
    # df = df.sample(n_test, replace=True)
    df = pd.concat([df1, df2])

ds = Dataset.from_pandas(df)
test_loader = DataLoader(ds, batch_size=bs)

# making the patches
# unfortunately, they are very prompt and tokenizer-specific
# has to account for different lengths of names and roles
if patch_tokens == 'precise':
    model_name_lower = model_name.lower()
    if method in ['das', 'bdas', 'intinv', 'zero-ablate']:
        if 'alpaca' in model_name_lower:
            if base_task == 'Admissions':
                dist_to_patch = 16
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'HireDec':
                dist_to_patch = 18
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'Hiring':
                dist_to_patch = 18
                patches = (dist_to_patch + np.arange(0, 4)).tolist()
            elif base_task == 'HireDecNames':
                dist_to_patch = 17
                patches = (dist_to_patch + np.arange(0, 4)).tolist()
            elif base_task == 'DiscrimEval':
                patches = np.arange(30, 60).tolist()
            elif base_task == 'RaceQA':
                dist_to_patch = 15
                patches = (dist_to_patch + np.arange(0, 4)).tolist()
            elif base_task == 'HiringRaceOffset':
                token_ids = tokenizer(ds['base'][:100], 
                                    padding=True, 
                                    return_tensors="pt").input_ids
                max_seq_len = token_ids.shape[1]
                dist_to_patch = max_seq_len - 37
                patches = (dist_to_patch + np.arange(0, 1)).tolist()

        elif 'mistral' in model_name_lower:
            if base_task == 'Admissions':
                dist_to_patch = 43
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'HireDec':
                dist_to_patch = 40
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'Hiring':
                dist_to_patch = 18
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'HireDecNames':
                dist_to_patch = 17
                patches = (dist_to_patch + np.arange(0, 4)).tolist()
            elif base_task == 'RaceQA':
                dist_to_patch = 15
                patches = (dist_to_patch + np.arange(0, 4)).tolist()
            elif base_task == 'HiringRaceOffset':
                token_ids = tokenizer(ds['base'][:100], 
                                    padding=True, 
                                    return_tensors="pt").input_ids
                max_seq_len = token_ids.shape[1]
                dist_to_patch = max_seq_len - 40
                patches = (dist_to_patch + np.arange(0, 3)).tolist()

        elif 'gemma' in model_name_lower:
            if base_task == 'Admissions':
                dist_to_patch = 13
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'HireDec':
                dist_to_patch = 14
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'Hiring':
                dist_to_patch = 15
                patches = (dist_to_patch + np.arange(0, 3)).tolist()
            elif base_task == 'HireDecNames':
                dist_to_patch = 15
                patches = (dist_to_patch + np.arange(0, 5)).tolist()
            elif base_task == 'RaceQA':
                dist_to_patch = 12
                patches = (dist_to_patch + np.arange(0, 4)).tolist()

    elif method == 'probing':
        num_pos = 5

        if 'llama-3' in model_name_lower:
            if base_task == 'Admissions':
                layer_start = 22
                layer_end = 30
                h_range = 7
                src_token_start = collect_token
                base_token_start = patch_token_positions[0]
                src_token_end = src_token_start + h_range
                base_token_end = base_token_start + h_range
            elif base_task == 'Hiring':
                layer_start = 18
                layer_end = 28
                h_range = 7
                src_token_start = collect_token
                base_token_start = patch_token_positions[0]
                src_token_end = src_token_start + h_range
                base_token_end = base_token_start + h_range
        
        if 'alpaca' in model_name_lower:
            if base_task == 'Admissions':
                # places to collect activations
                # base and source layers must be the same
                layer_start = 4
                layer_end = 14
                src_token_start = 62
                src_token_end = 71
                base_token_start = src_token_start
                base_token_end = src_token_end

            elif base_task == 'Hiring':
                layer_start = 8 # interventions work in the same layers
                layer_end = 16
                src_token_start = 17
                src_token_end = 29
                base_token_start = 22
                base_token_end = 25

        elif 'mistral' in model_name_lower:
            if base_task == 'Admissions':
                layer_start = 18
                layer_end = 26
                src_token_start = 96
                src_token_end = 101
                base_token_start = src_token_start
                base_token_end = src_token_end

            elif base_task == 'Hiring':
                layer_start = 16
                layer_end = 20
                src_token_start = 43
                src_token_end = 55
                base_token_start = 21
                base_token_end = 24

        elif 'gemma' in model_name_lower:
            layer_start = 2
            layer_end = 16
            h_range = 3
            src_token_start = collect_token
            base_token_start = patch_token_positions[0]
            src_token_end = src_token_start + h_range
            base_token_end = base_token_start + h_range
                
        patch_layers = random.choices(
            range(layer_start, layer_end+1), k=num_pos
        )

        base_token_pos = random.choices(
            range(base_token_start, base_token_end+1), k=num_pos
        )
        src_token_pos = random.choices(
            range(src_token_start, src_token_end+1), k=num_pos
        )
        # src_token_pos = base_token_pos

        save_path += "_collect"
        for layer, token in zip(patch_layers, src_token_pos):
            print(f"Collect position: {layer}.{token}")
            save_path += f"_{layer}.{token}"

        save_path += "_patch"
        for layer, token in zip(patch_layers, base_token_pos):
            print(f"Patch position: {layer}.{token}")
            save_path += f"_{layer}.{token}"

    elif method == 'random':
        num_pos = 5
        layer_start = 0
        layer_end = config.num_hidden_layers

        max_base_len = tokenizer(
            ds['base'][:100], padding=True, return_tensors="pt"
        ).input_ids.shape[1]

        max_src_len = tokenizer(
            ds['source'][:100], padding=True, return_tensors="pt"
        ).input_ids.shape[1]

        patch_layers = random.sample(
            range(layer_start, layer_end), num_pos
        )

        base_token_pos = random.sample(range(0, max_base_len), num_pos)
        src_token_pos = random.sample(range(0, max_src_len), num_pos)

        for layer, token in zip(patch_layers, src_token_pos):
            print(f"Collect position: {layer}.{token}")
            # save_path += f"_{layer}.{token}"

        for layer, token in zip(patch_layers, base_token_pos):
            print(f"Patch position: {layer}.{token}")

        # base_token_pos = patch_token_positions
        # src_token_pos = patch_token_positions

os.makedirs(save_path, exist_ok=True)

# Intervenable for activation collection
config_collect = pv.IntervenableConfig(
    [{
        "layer": collect_layer,
        "component": "block_output",
        "intervention_type": pv.CollectIntervention
    }],
)
vene_collect = pv.IntervenableModel(config_collect, llama)
vene_collect.set_device(device)
vene_collect.disable_model_gradients()

if args.method == "bdas":
    # Boundless rotated interchange intervention
    config_bdas = pv.IntervenableConfig(
        [{
            "layer": layer,
            "component": "block_output",
            "intervention_type": pv.BoundlessRotatedSpaceIntervention,
        }
            for layer in patch_layers
        ],
    )
    vene_bdas = load_alignment(align_path, config_bdas, llama, 
                            src_save_path=src_align_path)
    vene_bdas.set_device(device)
    vene_bdas.disable_model_gradients()

    vene = vene_bdas
    print("Using boundless DAS!")

elif args.method == "das":
    # Vanilla DAS
    config_das = pv.IntervenableConfig([
        {
            "layer": patch_layers[0],
            "component": 'block_output',
            "intervention_type": pv.RotatedSpaceIntervention,
        }
    ])
    vene_das = load_alignment(align_path, config_das, llama,
                              src_save_path=src_align_path, 
                              alignment_type=pv.RotatedSpaceIntervention)
    vene_das.set_device(device)
    vene_das.disable_model_gradients()

    vene = vene_das
    print("Using vanilla DAS!")

else:
    # Vanilla interchange intervention
    config_intinv = pv.IntervenableConfig(
        [{
            "layer": layer,
            "component": "block_output",
            "intervention_type": pv.VanillaIntervention,
        }
            for layer in patch_layers
        ],
    )
    vene_intinv = pv.IntervenableModel(config_intinv, llama)
    vene_intinv.set_device(device)
    vene_intinv.disable_model_gradients()

    vene = vene_intinv
    print("Using vanilla interchange!")

all_base_preds = []
all_ctf_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Intervening"):
        base_tokens = tokenizer(batch['base'], 
                                return_tensors='pt', 
                                padding=True).to(device)
        
        if use_right_padding:
            tokenizer.padding_side = 'right'
            src_tokens = tokenizer(batch['source'], 
                                return_tensors='pt', 
                                padding=True).to(device)
            tokenizer.padding_side = 'left'
        else:
            src_tokens = tokenizer(batch['source'], 
                                return_tensors='pt', 
                                padding=True).to(device)
        
        bs = base_tokens['input_ids'].shape[0]
        seq_len = base_tokens['input_ids'].shape[1]

        base = base_tokens.input_ids
        src = src_tokens.input_ids

        # breakpoint()

        if patch_tokens == 'custom':
            if patch_token_positions == None:
                raise ValueError("Token positions must be specified for \"custom\" patch ")
            patches = patch_token_positions
        elif patch_tokens == 'naive':
            patches = np.arange(0, seq_len).tolist()
        elif patch_tokens == 'random':
            patches = random.sample(range(0, seq_len), 5)

        num_pos = len(patches)

        if patches[0] < 0:
            patches_ = list(base.shape[1] + np.array(patches))
        else:
            patches_ = patches

        if intervention == 'interchange' and method in ["bdas", "das", "full"]:
            if collect_token < 0:
                collect_token_ = src.shape[1] + collect_token
            else:
                collect_token_ = collect_token
            
            intervenable_out = vene_collect(
                src_tokens,                                 
                unit_locations={"base": collect_token_},
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1] 
            activations = torch.concatenate(activations)

            src_activations = activations.unsqueeze(1)
            src_activations = src_activations.expand(-1, num_pos, -1)

            # breakpoint()

        elif intervention == 'interchange' and method in ['probing', 'random']:
            if base_token_pos[0] < 0:
                base_token_pos = list(base.shape[1] + np.array(base_token_pos))
            if src_token_pos[0] < 0:
                src_token_pos = list(src.shape[1] + np.array(src_token_pos))

            base_patches = [[[pos]] * bs for pos in base_token_pos]
            src_patches = [[[pos]] * bs for pos in src_token_pos]

            src_activations = None
            patches_ = (base_patches, src_patches)

        elif intervention == 'zero-ablate':
            src_activations = torch.zeros(bs, num_pos, config.hidden_size).to(device)
            
        elif intervention == 'neuron-avg':
            if collect_token < 0:
                collect_token_ = base.shape[1] + collect_token
            else:
                collect_token_ = collect_token

            intervenable_out = vene_collect(
                base_tokens,                                 
                unit_locations={"base": collect_token_},
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1] 
            activations = torch.concatenate(activations)

            acts_mean = activations.mean(dim=1).reshape(bs, 1, 1)
            src_activations = acts_mean.expand(-1, num_pos, config.hidden_size)
            
        elif intervention == 'noising':
            eps = 4
            if collect_token < 0:
                collect_token_ = base.shape[1] + collect_token
            else:
                collect_token_ = collect_token

            intervenable_out = vene_collect(
                base_tokens,                                 
                unit_locations={"base": collect_token_},
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1] 
            activations = torch.concatenate(activations).unsqueeze(1)
            acts_std = torch.std(activations)

            src_activations = activations.expand(-1, num_pos, -1)
            noise = torch.normal(mean=0, std=eps*acts_std, 
                                 size=src_activations.shape).to(device)
            src_activations = (src_activations + noise)
            
        elif intervention == 'var-avg':
            if collect_token < 0:
                # collect_token_ = base.shape[1] + collect_token
                collect_token_ = src.shape[1] + collect_token
            else:
                collect_token_ = collect_token

            intervenable_out = vene_collect(
                # base_tokens,
                src_tokens,                                 
                unit_locations={"base": collect_token_},
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1] 
            activations = torch.concatenate(activations)

            acts_mean = activations.mean(dim=0).reshape(1, 1, config.hidden_size)
            src_activations = acts_mean.expand(bs, num_pos, -1)

            # print("Here!")

        elif intervention == 'prompting':
            debias_base = [prompt.replace("\n\nAnswer:\n", debias_prompt) 
                            for prompt in batch['base']]
            base_tokens = tokenizer(debias_base, 
                                return_tensors='pt', 
                                padding=True).to(device)
            
            src_tokens = None
            src_activations = None
            patches_ = []
            # breakpoint()

        if not generate:
            base_outputs, ctf_outputs = vene(
                base_tokens,
                src_tokens,
                source_representations = src_activations, # (bs, num_pos, hidden)
                output_original_output = True,
                unit_locations = {"sources->base": patches_},
            )

            base_logits = base_outputs.logits[:, -1]
            ctf_logits = ctf_outputs.logits[:, -1]

            base_preds = base_logits.argmax(dim=-1).cpu().tolist()
            ctf_preds = ctf_logits.argmax(dim=-1).cpu().tolist()
        else: # Note: not tested
            base_outputs, ctf_outputs = vene.generate(
                base_tokens,
                src_tokens,
                source_representations = src_activations,
                max_length = seq_len + 10,
                output_original_output = True,
                intervene_on_prompt = True,
                unit_locations = {"base": patches_},
            )

            base_outputs = base_outputs[:, seq_len:]
            ctf_outputs = ctf_outputs[:, seq_len:]

            base_preds = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
            ctf_preds = tokenizer.batch_decode(ctf_outputs, skip_special_tokens=True)

            base_preds = [pred.strip() for pred in base_preds]
            ctf_preds = [pred.strip() for pred in ctf_preds]

        all_base_preds += base_preds
        all_ctf_preds += ctf_preds

df['base_pred'] = all_base_preds
df['ctf_pred'] = all_ctf_preds
df.to_csv(os.path.join(save_path, "preds.csv"), index=False)

acc = accuracy_score(df['ctf_pred'], df['src_label'])
with open(os.path.join(save_path, "iia.txt"), 'w') as fw:
    fw.write(f"Test IIA: {acc:.4f}")
