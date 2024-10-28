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
from utils import get_bdas_params, load_alignment


# """
# Load a trained BoundlessRotatedSpace alignment. Assumes the user is only
# loading in one alignment potentially across multiple layers.
# """
# def load_alignment(save_path, config, model):
#     # We assume the model is saved with these two files
#     model_path = os.path.join(save_path, "model.pt")
#     model_params_path = os.path.join(save_path, "model_params.pt")

#     intervenable = pv.IntervenableModel(config, model)
#     intervenable.load_state_dict(torch.load(model_path))
#     intervention_params = pv.BoundlessRotatedSpaceIntervention(
#         embed_dim=model.config.hidden_size
#     )
#     intervention_params.load_state_dict(torch.load(model_params_path))

#     keys = list(intervenable.representations.keys())
#     for key in keys:
#         hook = intervenable.interventions[key][1]
#         intervenable.interventions[key] = (intervention_params, hook)
    
#     return intervenable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input arguments for cross-task intervention on HireDec.")

    parser.add_argument('--model_name', type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')
    parser.add_argument("--alignment_path", 
                        help="""Path to the directory 
                        containing the saved alignment.""")
    parser.add_argument("--dataset_path", 
                        help="""Path to the directory containing
                        the counterfactual dataset files.""")
    parser.add_argument("--base_task",
                        choices=['Admissions', 'HireDec', 'HireDecEval', 'HireDecNames', 'DiscrimEval', 'RaceQA'],
                        default="Admissions")

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--generate', action='store_true')

    parser.add_argument('--source_reduction', type=str, 
                        choices=["selection", "mean", "zero-ablation"], 
                        default="selection")
    
    parser.add_argument('--method', 
                        choices=["das", "probing", "random"], default="das")

    parser.add_argument('--collect_layer', type=int, default=2, 
                        help='Layer to collect activations from. This option only applies to DAS.')
    parser.add_argument('--collect_token', type=int, default=17, 
                        help='Position to collect activations from. This option only applies to DAS.')

    parser.add_argument('--patch_start', type=int, default=2, 
                        help='Start of patch layers range')
    parser.add_argument('--patch_end', type=int, default=4, 
                        help='End of patch layers range')
    
    parser.add_argument('--patch_tokens', type=str, 
                        choices=["naive", "precise", "random", "custom"], default="precise", 
                        help='Type of patch tokens')
    parser.add_argument('--patch_token_positions', nargs='+', type=int)
    
    parser.add_argument('--concept_subspace', type=str, 
                        choices=["naive", "aligned"], default="aligned", 
                        help='Concept direction for intervention')

    parser.add_argument("--save_path", default="./", help="Path to the save directory")

    args = parser.parse_args()
    return args


args = parse_args()

bs = args.batch_size
n_test = args.n_test
src_reduce = args.source_reduction
generate = args.generate

method = args.method
collect_layer = args.collect_layer
collect_token = args.collect_token

patch_layers = range(args.patch_start, args.patch_end+1)
patch_tokens = args.patch_tokens
patch_token_positions = args.patch_token_positions
concept_subspace = args.concept_subspace

ds_path = args.dataset_path
base_task = args.base_task
align_path = args.alignment_path
save_path = args.save_path

model_name = args.model_name
device = 'cuda'

os.makedirs(save_path, exist_ok=True)

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
if n_test != -1:
    df = df.sample(n_test, replace=True)

ds = Dataset.from_pandas(df)
test_loader = DataLoader(ds, batch_size=bs)

# making the patches
# unfortunately, they are very prompt and tokenizer-specific
# has to account for different lengths of names and roles
model_name_lower = model_name.lower()
if method == 'das':
    if 'alpaca' in model_name_lower:
        if base_task == 'Admissions':
            dist_to_patch = 16
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
        elif base_task == 'HireDec':
            dist_to_patch = 18
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
        elif base_task == 'HireDecEval':
            dist_to_patch = 18
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
            # patches = (dist_to_patch + np.arange(0, 7)).tolist()
        elif base_task == 'HireDecNames':
            dist_to_patch = 17
            patches = (dist_to_patch + np.arange(0, 4)).tolist()
        elif base_task == 'DiscrimEval':
            patches = np.arange(30, 60).tolist()
        elif base_task == 'RaceQA':
            dist_to_patch = 15
            patches = (dist_to_patch + np.arange(0, 4)).tolist()

    elif 'mistral' in model_name_lower:
        if base_task == 'Admissions':
            dist_to_patch = 43
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
        elif base_task == 'HireDec':
            dist_to_patch = 40
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
        elif base_task == 'HireDecEval':
            dist_to_patch = 18
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
            # patches = (dist_to_patch + np.arange(0, 6)).tolist()
        elif base_task == 'HireDecNames':
            dist_to_patch = 17
            patches = (dist_to_patch + np.arange(0, 4)).tolist()
        elif base_task == 'RaceQA':
            dist_to_patch = 15
            patches = (dist_to_patch + np.arange(0, 4)).tolist()

    elif 'gemma' in model_name_lower:
        if base_task == 'Admissions':
            dist_to_patch = 13
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
        elif base_task == 'HireDec':
            dist_to_patch = 14
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
        elif base_task == 'HireDecEval':
            dist_to_patch = 15
            patches = (dist_to_patch + np.arange(0, 3)).tolist()
            # patches = (dist_to_patch + np.arange(0, 4)).tolist()
        elif base_task == 'HireDecNames':
            dist_to_patch = 15
            patches = (dist_to_patch + np.arange(0, 5)).tolist()
        elif base_task == 'RaceQA':
            dist_to_patch = 12
            patches = (dist_to_patch + np.arange(0, 4)).tolist()

elif method == 'probing':
    num_pos = 5
    
    if 'alpaca' in model_name_lower:
        if base_task == 'Admissions':
            # places to collect activations
            # base and source layers must be the same
            layer_start = 4
            layer_end = 16
            src_token_start = 17
            src_token_end = 29
            base_token_start = src_token_start
            base_token_end = src_token_end

        elif base_task == 'HireDecEval':
            layer_start = 8 # interventions work in the same layers
            layer_end = 16
            src_token_start = 17
            src_token_end = 29
            base_token_start = 22
            base_token_end = 25

    elif 'mistral' in model_name_lower:
        if base_task == 'Admissions':
            layer_start = 6
            layer_end = 30
            src_token_start = 43
            src_token_end = 55
            base_token_start = src_token_start
            base_token_end = src_token_end

        elif base_task == 'HireDecEval':
            layer_start = 16
            layer_end = 20
            src_token_start = 43
            src_token_end = 55
            base_token_start = 21
            base_token_end = 24

    elif 'gemma' in model_name_lower:
        if base_task == 'Admissions':
            layer_start = 6
            layer_end = 16
            src_token_start = 14
            src_token_end = 24
            base_token_start = src_token_start
            base_token_end = src_token_end

        elif base_task == 'HireDecEval':
            layer_start = 6
            layer_end = 12
            src_token_start = 14
            src_token_end = 24
            base_token_start = 10
            base_token_end = 10
            
    patch_layers = random.choices(
        range(layer_start, layer_end+1), k=num_pos
    )

    base_token_pos = random.choices(
        range(base_token_start, base_token_end+1), k=num_pos
    )
    src_token_pos = random.choices(
        range(src_token_start, src_token_end+1), k=num_pos
    )

    for layer, token in zip(patch_layers, base_token_pos):
        print(f"Collect position: {layer}.{token}")
        save_path += f"_{layer}.{token}"
    os.makedirs(save_path, exist_ok=True)

    concept_subspace = 'naive' # overwriting

elif method == 'random':
    num_pos = 5
    layer_start = 0
    layer_end = config.num_hidden_layers

    max_seq_len = tokenizer(
        ds['base'][:100], padding=True, return_tensors="pt"
    ).input_ids.shape[1]

    patch_layers = random.sample(
        range(layer_start, layer_end), num_pos
    )
    patch_token_positions = random.sample(range(0, max_seq_len), num_pos)

    for layer, token in zip(patch_layers, patch_token_positions):
        print(f"Collect position: {layer}.{token}")
        save_path += f"_{layer}.{token}"
    os.makedirs(save_path, exist_ok=True)

    base_token_pos = patch_token_positions
    src_token_pos = patch_token_positions
    concept_subspace = 'naive'


# Intervenable for activation collection
# config_collect = pv.IntervenableConfig(
#     [{
#         "layer": layer,
#         "component": "block_output",
#         "intervention_type": pv.CollectIntervention
#     }
#         for layer in collect_layers
#     ],
# )
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

# Rotated interchange intervention
config_bdas = pv.IntervenableConfig(
    [{
        "layer": layer,
        "component": "block_output",
        "intervention_type": pv.BoundlessRotatedSpaceIntervention,
    }
        for layer in patch_layers
    ],
)
vene_bdas = load_alignment(align_path, config_bdas, llama)
vene_bdas.set_device(device)
vene_bdas.disable_model_gradients()
intervention, _, _ = get_bdas_params(vene_bdas)

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


# if src_reduce == "mean":
#     all_src_acts = []
#     for batch in tqdm(test_loader, desc="Processing source inputs"):
#         batch_tokens = tokenizer(batch['source'], 
#                                 return_tensors="pt", 
#                                 padding=True).to(device)
        
#         intervenable_out = vene_collect(
#             batch_tokens,                                 
#             unit_locations={"base": collect_tokens},       
#         )
#         # intervenable_out is ((_, activations), _)
#         src_activations = intervenable_out[0][1]
#         src_activations = torch.concatenate(src_activations)
#         all_src_acts.append(src_activations)

#     all_src_acts = torch.concatenate(all_src_acts)
#     assert all_src_acts.shape == (len(ds['base']), llama.config.hidden_size)

#     src_rep = all_src_acts.mean(dim=0)

# elif src_reduce == "zero-ablation":
#     src_rep = torch.zeros(llama.config.hidden_size)


if concept_subspace == "naive":
    vene = vene_intinv
    print("Using vanilla interchange!")
elif concept_subspace == "aligned":
    vene = vene_bdas
    print("Using boundless DAS!")

all_base_preds = []
all_ctf_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Intervening"):
        base_prompts = batch['base']
        base_tokens = tokenizer(base_prompts, 
                                return_tensors='pt', 
                                padding=True).to(device)
        src_tokens = tokenizer(batch['source'], 
                               return_tensors='pt', 
                               padding=True).to(device)
        
        bs = base_tokens['input_ids'].shape[0]
        seq_len = base_tokens['input_ids'].shape[1]

        if args.method == 'das':
            # default patch is "precise"
            if patch_tokens == 'custom':
                if patch_token_positions == None:
                    raise ValueError("Token positions must be specified for \"custom\" patch ")
                patches = patch_token_positions
            elif patch_tokens == 'naive':
                patches = np.arange(0, seq_len).tolist()
            elif patch_tokens == 'random':
                patches = random.sample(range(0, seq_len), 5)

            num_pos = len(patches)

            intervenable_out = vene_collect(
                src_tokens,                                 
                unit_locations={"base": collect_token},
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1] 
            activations = torch.concatenate(activations)

            src_activations = activations.unsqueeze(1)
            src_activations = src_activations.expand(-1, num_pos, -1)

            if not generate:
                base_outputs, ctf_outputs = vene(
                    base_tokens,
                    source_representations = src_activations, # (bs, num_pos, hidden)
                    output_original_output = True,
                    unit_locations = {"sources->base": patches},
                )

                base_logits = base_outputs.logits[:, -1]
                ctf_logits = ctf_outputs.logits[:, -1]

                base_preds = base_logits.argmax(dim=-1).cpu().tolist()
                ctf_preds = ctf_logits.argmax(dim=-1).cpu().tolist()
            else:
                base_outputs, ctf_outputs = vene.generate(
                    base_tokens,
                    source_representations = src_activations,
                    max_length = seq_len + 10,
                    output_original_output = True,
                    intervene_on_prompt = True,
                    unit_locations = {"base": patches},
                )

                base_outputs = base_outputs[:, seq_len:]
                ctf_outputs = ctf_outputs[:, seq_len:]

                base_preds = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
                ctf_preds = tokenizer.batch_decode(ctf_outputs, skip_special_tokens=True)

                base_preds = [pred.strip() for pred in base_preds]
                ctf_preds = [pred.strip() for pred in ctf_preds]

        elif args.method == 'probing' or args.method == 'random':
            base_patches = [[[pos]] * bs for pos in base_token_pos]
            src_patches = [[[pos]] * bs for pos in src_token_pos]

            if generate:
                print("Generation for probing and random interventions is not supported yet.")
                pass
            else:
                base_outputs, ctf_outputs = vene(
                    base_tokens,
                    src_tokens,
                    output_original_output = True,
                    unit_locations = {
                        "sources->base": (src_patches, base_patches)
                    },
                )

                base_logits = base_outputs.logits[:, -1]
                ctf_logits = ctf_outputs.logits[:, -1]

                base_preds = base_logits.argmax(dim=-1).cpu().tolist()
                ctf_preds = ctf_logits.argmax(dim=-1).cpu().tolist()

        all_base_preds += base_preds
        all_ctf_preds += ctf_preds

        # # default patch is "precise"
        # if patch_tokens == 'custom':
        #     if patch_token_positions == None:
        #         raise ValueError("Token positions must be specified for \"custom\" patch ")
        #     else:
        #         patches = patch_token_positions
        # elif patch_tokens == 'naive':
        #     patches = np.arange(0, seq_len).tolist()
        # elif patch_tokens == 'random':
        #     patches = np.random.randint(0, seq_len, 5).tolist()

        # num_pos = len(patches)

        # if src_reduce == "selection":
        #     src_tokens = tokenizer(batch['source'], 
        #                            return_tensors='pt', 
        #                            padding=True).to(device)

        #     intervenable_out = vene_collect(
        #         src_tokens,                                 
        #         unit_locations={"base": collect_tokens},          
        #     )
        #     # intervenable_out is ((_, activations), _)
        #     activations = intervenable_out[0][1]
        #     activations = torch.concatenate(activations)

        #     if args.method == 'das':
        #         src_activations = activations.unsqueeze(1)
        #         src_activations = activations.expand(-1, num_pos, -1)
        #     else:
        #         # src_activations = activations.reshape(
        #         #     bs, num_pos, -1
        #         # )
        #         src_activations = activations.reshape(
        #             num_pos, bs, 1, -1
        #         )
        #         src_activations = einops.rearrange(
        #             src_activations, 
        #             'num_int bs num_pos hidden -> bs num_int num_pos hidden'
        #         ) # num_pos = num_int for random-layer interventions
        # else:
        #     src_activations = (
        #         src_rep.reshape(1, 1, -1)
        #         .expand(len(base_prompts), num_pos, -1) 
        #     ) # shape (bs, num_pos, hidden)

        # if generate:
        #     base_outputs, ctf_outputs = vene.generate(
        #         base_tokens,
        #         source_representations = src_activations,
        #         max_length = seq_len + 10,
        #         output_original_output = True,
        #         intervene_on_prompt = True,
        #         unit_locations = {"base": patches},
        #     )

        #     base_outputs = base_outputs[:, seq_len:]
        #     ctf_outputs = ctf_outputs[:, seq_len:]

        #     base_preds = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
        #     ctf_preds = tokenizer.batch_decode(ctf_outputs, skip_special_tokens=True)

        #     base_preds = [pred.strip() for pred in base_preds]
        #     ctf_preds = [pred.strip() for pred in ctf_preds]

        # else:
        #     base_outputs, ctf_outputs = vene(
        #         base_tokens,
        #         source_representations = src_activations, # (bs, num_pos, hidden)
        #         output_original_output = True,
        #         unit_locations = {"sources->base": patches},
        #     )

        #     base_logits = base_outputs.logits[:, -1]
        #     ctf_logits = ctf_outputs.logits[:, -1]

        #     base_preds = base_logits.argmax(dim=-1).cpu().tolist()
        #     ctf_preds = ctf_logits.argmax(dim=-1).cpu().tolist()

        # all_base_preds += base_preds
        # all_ctf_preds += ctf_preds

df['base_pred'] = all_base_preds
df['ctf_pred'] = all_ctf_preds
df.to_csv(os.path.join(save_path, "preds.csv"), index=False)

acc = accuracy_score(df['ctf_pred'], df['src_label'])
with open(os.path.join(save_path, "iia.txt"), 'w') as fw:
    fw.write(f"Test IIA: {acc:.4f}")
