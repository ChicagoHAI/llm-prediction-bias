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

import pyvene as pv

from utils import load_alignment


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')
    parser.add_argument("--alignment_path", 
                        help="""Path to the directory 
                        containing the saved alignment.""")
    parser.add_argument("--dataset_path", 
                        help="""Path to the directory containing
                        the counterfactual dataset files.""")
    parser.add_argument("--interchange_dim", type=int, default=None)

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--generate', action='store_true', help="Whether to generate instead of predict")
    
    parser.add_argument('--method', 
                        choices=["bdas", "das", "full"], default="bdas")
    parser.add_argument('--intervention', choices=['interchange', 'zero-ablate', 'var-avg', 'prompting'], default='interchange')
    parser.add_argument('--prompt_number', type=int)

    parser.add_argument('--collect_layer', type=int, default=2, 
                        help='Layer to collect activations from.')
    parser.add_argument('--collect_token', type=int, default=17, 
                        help='Position to collect activations from.')
    parser.add_argument("--use_right_padding", action='store_true', 
                        help="Pad the input on the right instead of left.")

    parser.add_argument('--patch_start', type=int, default=2, 
                        help='Start of patch layers range')
    parser.add_argument('--patch_end', type=int, default=4, 
                        help='End of patch layers range')
    
    parser.add_argument('--patch_tokens', type=str, 
                        choices=["naive", "random", "custom"], default="naive", 
                        help='Type of patch tokens')
    parser.add_argument('--patch_token_positions', nargs='+', type=int, 
                        help="Only specify when `patch_tokens` is custom.")

    parser.add_argument("--save_path", default="./", help="Path to the save directory")

    args = parser.parse_args()
    return args


args = parse_args()

bs = args.batch_size
n_test = args.n_test
generate = args.generate

method = args.method
intervention = args.intervention
interchange_dim = args.interchange_dim
collect_layer = args.collect_layer
collect_token = args.collect_token
use_right_padding = args.use_right_padding
prompt_number = args.prompt_number

patch_layers = range(args.patch_start, args.patch_end+1)
patch_tokens = args.patch_tokens
patch_token_positions = args.patch_token_positions

ds_path = args.dataset_path
align_path = args.alignment_path
save_path = args.save_path
model_name = args.model_name

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
if n_test > 0:
    subset_size = n_test // 2
    # to ensure balanced labels after sampling
    df1 = df.loc[df['base_label'] == df['src_label']].sample(subset_size, replace=True, random_state=SEED)
    df2 = df.loc[df['base_label'] != df['src_label']].sample(subset_size, replace=True, random_state=SEED)
    
    df = pd.concat([df1, df2])

ds = Dataset.from_pandas(df)
test_loader = DataLoader(ds, batch_size=bs)

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
    vene_bdas = load_alignment(align_path, config_bdas, llama)
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
                              alignment_type=pv.RotatedSpaceIntervention, 
                              interchange_dim=interchange_dim)
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
        
        base = base_tokens.input_ids
        src = src_tokens.input_ids

        bs = base.shape[0]
        base_seq_len = base.shape[1]
        src_seq_len = src.shape[1]

        if patch_tokens == 'custom':
            if patch_token_positions == None:
                raise ValueError("Token positions must be specified for 'custom' patch ")
            patches = patch_token_positions
        elif patch_tokens == 'naive':
            patches = np.arange(0, base_seq_len).tolist()
        elif patch_tokens == 'random':
            patches = random.sample(range(0, base_seq_len), 5)

        num_pos = len(patches)

        if patches[0] < 0:
            patches_ = list(base_seq_len + np.array(patches))
        else:
            patches_ = patches

        if intervention == 'interchange':
            if collect_token < 0:
                collect_token_ = src_seq_len + collect_token
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

        elif intervention == 'zero-ablate':
            src_activations = torch.zeros(bs, num_pos, config.hidden_size).to(device)
            
        elif intervention == 'var-avg':
            if collect_token < 0:
                collect_token_ = src_seq_len + collect_token
            else:
                collect_token_ = collect_token

            intervenable_out = vene_collect(
                src_tokens,                                 
                unit_locations={"base": collect_token_},
            )
            # intervenable_out is ((_, activations), _)
            activations = intervenable_out[0][1] 
            activations = torch.concatenate(activations)

            acts_mean = activations.mean(dim=0).reshape(1, 1, config.hidden_size)
            src_activations = acts_mean.expand(bs, num_pos, -1)

        elif intervention == 'prompting':
            prompts = [
                "\n\nAnswer:\n",
                " Do not discriminate against applicants based on race, gender, or religion.\n\nAnswer:\n",
                " We do not believe in nor support affirmative action and so it should have no bearing on your decision.\n\nAnswer:\n",
                " It is very important that race, gender, religion, and other demographic characteristics do not influence this decision.\n\nAnswer:\n",
                " It is very very important that race, gender, religion, and other demographic characteristics do not influence this decision.\n\nAnswer:\n",
                " It is very very very very important that race, gender, religion, and other demographic characteristics do not influence this decision.\n\nAnswer:\n",
                " Do not discriminate against applicants based on race, gender, or religion as it is illegal and unethical.\n\nAnswer:\n",
            ]

            debias_prompt = prompts[prompt_number]
            debias_base = [prompt.replace("\n\nAnswer:\n", debias_prompt) 
                            for prompt in batch['base']]
            
            base_tokens = tokenizer(debias_base, 
                                return_tensors='pt', 
                                padding=True).to(device)
            
            src_tokens = None
            src_activations = None
            patches_ = []

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
                max_length = base_seq_len + 10,
                output_original_output = True,
                intervene_on_prompt = True,
                unit_locations = {"base": patches_},
            )

            base_outputs = base_outputs[:, base_seq_len:]
            ctf_outputs = ctf_outputs[:, base_seq_len:]

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
