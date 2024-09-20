import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import argparse

sys.path.append('../pyvene/')
import pyvene as pv

from eval_alignment import load_alignment
from utils import get_bdas_params
from make_ctf_dataset import format_label


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
                        choices=['Admissions', 'HireDec', 'HireDecEval', 'HireDecNames', 'DiscrimEval'],
                        default="Admissions")

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--source_reduction', type=str, 
                        choices=["selection", "mean", "zero-ablation"], 
                        default="selection")

    parser.add_argument('--collect_layer', type=int, default=2, 
                        help='Layer to collect activations from')
    parser.add_argument('--collect_pos', type=int, default=17, 
                        help='Position to collect activations from')

    parser.add_argument('--patch_start', type=int, default=2, 
                        help='Start of patch layers range')
    parser.add_argument('--patch_end', type=int, default=4, 
                        help='End of patch layers range')
    
    parser.add_argument('--patch_tokens', type=str, 
                        choices=["naive", "precise", "random"], default="precise", 
                        help='Type of patch tokens')
    parser.add_argument('--token_positions', nargs='+', type=int)
    
    parser.add_argument('--concept_subspace', type=str, 
                        choices=["naive", "aligned"], default="aligned", 
                        help='Concept direction for intervention')

    parser.add_argument("--save_path", default="./", help="Path to the save directory")

    args = parser.parse_args()
    return args


args = parse_args()

bs = args.batch_size
src_reduce = args.source_reduction

collect_layer = args.collect_layer
collect_pos = args.collect_pos

patch_layers = range(args.patch_start, args.patch_end+1)
patch_tokens = args.patch_tokens
token_positions = args.token_positions
concept_subspace = args.concept_subspace

ds_path = args.dataset_path
base_task = args.base_task
align_path = args.alignment_path
save_path = args.save_path

model_name = args.model_name
device = 'cuda:1'

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

# Intervenable for activation collection
config_collect = pv.IntervenableConfig([
    {
        "layer": collect_layer,
        "component": "block_output",
        "intervention_type": pv.CollectIntervention
    }
])
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

df = pd.read_csv(os.path.join(ds_path, 'test.csv')).sample(800, replace=True)

ds = Dataset.from_pandas(df)
test_loader = DataLoader(ds, batch_size=bs)

if src_reduce == "mean":
    all_src_acts = []
    for batch in tqdm(test_loader, desc="Processing source inputs"):
        batch_tokens = tokenizer(batch['source'], 
                                return_tensors="pt", 
                                padding=True).to(device)
        
        intervenable_out = vene_collect(
            batch_tokens,                                 
            unit_locations={"base": collect_pos},                                       
        )
        # intervenable_out is ((_, activations), _)
        src_activations = intervenable_out[0][1]
        src_activations = torch.concatenate(src_activations)
        all_src_acts.append(src_activations)

    all_src_acts = torch.concatenate(all_src_acts)
    assert all_src_acts.shape == (len(ds['base']), llama.config.hidden_size)

    src_rep = all_src_acts.mean(dim=0)

elif src_reduce == "zero-ablation":
    src_rep = torch.zeros(llama.config.hidden_size)

# making the "precise" patch
# unfortunately, it is very prompt and tokenizer-specific
# has to account for different lengths of names and roles

model_name_lower = model_name.lower()
if 'alpaca' in model_name_lower:
    if base_task == 'Admissions':
        dist_to_patch = 16
        patches = (dist_to_patch + np.arange(0, 3)).tolist()
    elif base_task == 'HireDec' or base_task == 'HireDecEval':
        dist_to_patch = 18
        patches = (dist_to_patch + np.arange(0, 3)).tolist()
    elif base_task == 'HireDecNames':
        dist_to_patch = 17
        patches = (dist_to_patch + np.arange(0, 4)).tolist()
    elif base_task == 'DiscrimEval':
        patches = np.arange(30, 60).tolist()

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
    elif base_task == 'HireDecNames':
        dist_to_patch = 17
        patches = (dist_to_patch + np.arange(0, 4)).tolist()

elif 'gemma' in model_name_lower:
    if base_task == 'Admissions':
        dist_to_patch = 14
        patches = (dist_to_patch + np.arange(0, 3)).tolist()
    elif base_task == 'HireDec' or base_task == 'HireDecEval':
        dist_to_patch = 15
        patches = (dist_to_patch + np.arange(0, 3)).tolist()
    elif base_task == 'HireDecNames':
        dist_to_patch = 15
        patches = (dist_to_patch + np.arange(0, 5)).tolist()


if concept_subspace == "naive":
    vene = vene_intinv
elif concept_subspace == "aligned":
    vene = vene_bdas

all_base_preds = []
all_ctf_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Intervening"):
        base_prompts = batch['base']
        base_tokens = tokenizer(base_prompts, 
                                return_tensors='pt', 
                                padding=True).to(device)
        
        seq_len = base_tokens['input_ids'].shape[1]

        if token_positions:
            patches = token_positions
        elif patch_tokens == 'naive':
            patches = np.arange(0, seq_len).tolist()
        elif patch_tokens == 'random':
            patches = np.random.randint(0, seq_len, len(patches)).tolist()
        num_pos = len(patches)

        if src_reduce == "selection":
            src_tokens = tokenizer(batch['source'], 
                                   return_tensors='pt', 
                                   padding=True).to(device)

            intervenable_out = vene_collect(
                src_tokens,                                 
                unit_locations={"base": collect_pos},                                       
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1]
            activations = torch.concatenate(activations).unsqueeze(1)
            src_activations = activations.expand(-1, num_pos, -1)
        else:
            src_activations = (
                src_rep.reshape(1, 1, -1)
                .expand(len(base_prompts), num_pos, -1) 
            ) # shape (bs, num_pos, hidden)

        if base_task != "DiscrimEval":
            base_outputs, ctf_outputs = vene(
                base_tokens,
                source_representations = src_activations,
                output_original_output = True,
                unit_locations = {"sources->base": patches},
            )

            base_logits = base_outputs.logits[:, -1]
            base_preds = base_logits.argmax(dim=-1).cpu().numpy()
            all_base_preds.append(base_preds)

            ctf_logits = ctf_outputs.logits[:, -1]
            ctf_preds = ctf_logits.argmax(dim=-1).cpu().numpy()
            all_ctf_preds.append(ctf_preds)
        else:
            base_outputs, ctf_outputs = vene.generate(
                base_tokens,
                source_representations = src_activations,
                max_length = seq_len + 10,
                output_original_output = True,
                intervene_on_prompt = True,
                unit_locations = {"base": patches},
            )

            base_gen = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
            ctf_gen = tokenizer.batch_decode(ctf_outputs, skip_special_tokens=True)

            # note: outdated, the Yes and No token indices vary between models
            base_preds = [8241 if "Yes" in gen else 3782 for gen in base_gen]
            ctf_preds = [8241 if "Yes" in gen else 3782 for gen in ctf_gen]

            all_base_preds.append(np.array(base_preds))
            all_ctf_preds.append(np.array(ctf_preds))

all_base_preds = np.concatenate(all_base_preds)
all_ctf_preds = np.concatenate(all_ctf_preds)

df['base_pred'] = all_base_preds
df['ctf_pred'] = all_ctf_preds
df.to_csv(os.path.join(save_path, "preds.csv"), index=False)

acc = accuracy_score(df['ctf_pred'], df['src_label'])
with open(os.path.join(save_path, "iia.txt"), 'w') as fw:
    fw.write(f"Test IIA: {acc:.4f}")
