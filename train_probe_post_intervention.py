import argparse
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import einops

import sys
sys.path.append('../pyvene/')
import pyvene as pv

from train_probe import LinearProbe, race_to_label, get_race
from eval_alignment import load_alignment


def process_dataset(dataset):
    # 12 is where race is in the Admissions prompt
    dataset = dataset.map(lambda x: {'base_race': get_race(12, x['base'])})
    dataset = dataset.map(lambda x: {'base_label': race_to_label(x['base_race'])})
    dataset = dataset.map(lambda x: {'src_race': get_race(12, x['source'])})
    dataset = dataset.map(lambda x: {'src_label': race_to_label(x['src_race'])})
    
    dataset = dataset.select_columns(['base','source','base_label','src_label'])
    return dataset

parser = argparse.ArgumentParser()

parser.add_argument("--make_acts_dataset", action="store_true")
parser.add_argument("--model_name", type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')

parser.add_argument("--alignment_path", type=str)
parser.add_argument("--collect_layer", type=int)
parser.add_argument("--collect_pos", type=int)

parser.add_argument("--dataset_size", type=int)
parser.add_argument("--train_dev_split", nargs='+', type=float)

parser.add_argument("--horizontal_start", type=int, default=0)
parser.add_argument("--horizontal_end", type=int, default=50)
parser.add_argument("--horizontal_step", type=int, default=1)

parser.add_argument("--vertical_start", type=int, default=0)
parser.add_argument("--vertical_end", type=int, default=-1)
parser.add_argument("--vertical_step", type=int, default=1)

parser.add_argument("--extra_steps", 
                    help="""The number of steps before {h_pos} to search.""", 
                    default=4, type=int)

parser.add_argument("--dataset_path", type=str)
parser.add_argument("--acts_save_path", default="./")
parser.add_argument("--results_save_path", type=str, default="./")

args = parser.parse_args()

make_acts_dataset = args.make_acts_dataset
model_name = args.model_name
align_path = args.alignment_path
layer = args.collect_layer
token_ind = args.collect_pos

ds_path = args.dataset_path
ds_size = args.dataset_size

h_start = args.horizontal_start
h_end = args.horizontal_end
h_step = args.horizontal_step
num_extra_steps = args.extra_steps

v_start = args.vertical_start
v_end = args.vertical_end
v_step = args.vertical_step

train_dev_split = args.train_dev_split
acts_save_path = args.acts_save_path
results_save_path = args.results_save_path

os.makedirs(acts_save_path, exist_ok=True)
os.makedirs(results_save_path, exist_ok=True)

bs = 32
num_epochs = 20
device = 'cuda:1'

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset('csv', data_files={
    'train': os.path.join(ds_path, 'train.csv'),
})

ds = ds['train'].shuffle(seed=42).select(range(ds_size))
ds = process_dataset(ds)
ds_loader = DataLoader(ds, batch_size=bs)

if make_acts_dataset:
    llama = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,  # save memory
    )
    _ = llama.to(device)
    _ = llama.eval()

    llama.config.output_hidden_states = True
    config_bdas = pv.IntervenableConfig([
        {
            "layer": layer,
            "component": 'block_output',
            "intervention_type": pv.BoundlessRotatedSpaceIntervention,
        }
    ])
    vene_bdas = load_alignment(align_path, config_bdas, llama)
    vene_bdas.set_device(device)
    vene_bdas.disable_model_gradients()

    acts = []
    with torch.no_grad():
        for batch in tqdm(ds_loader, desc="Getting train activations"):
            base_tokens = tokenizer(batch['base'], 
                                return_tensors="pt", 
                                padding=True).to(device)
            source_tokens = tokenizer(batch['source'], 
                                return_tensors="pt", 
                                padding=True).to(device)


            _, ctf_outputs = vene_bdas(
                                base_tokens,
                                [source_tokens],
                                {"sources->base": token_ind},
                            )

            hidden_states = [acts.unsqueeze(0) 
                            for acts in ctf_outputs.hidden_states]
            hidden_states = torch.concatenate(hidden_states)

            if hidden_states.shape[2] == 124: # length of Admissions prompt
                acts.append(hidden_states[:, :, :-1, :].cpu())
            else:
                acts.append(hidden_states.cpu())

    acts = torch.concatenate(acts, dim=1)
    acts = einops.rearrange(
        acts, 
        "layers samples tokens hidden_size -> samples layers tokens hidden_size" 
    )

    # making the train-dev-test split
    train_frac, dev_frac = train_dev_split
    train_end = int(train_frac * ds_size)
    dev_end = int((train_frac + dev_frac) * ds_size)

    train_acts = acts[:train_end, :, :, :]
    dev_acts = acts[train_end:dev_end, :, :, :]
    test_acts = acts[dev_end:, :, :, :]

    labels = torch.tensor(ds["base_label"])
    train_labels = labels[:train_end]
    dev_labels = labels[train_end:dev_end]
    test_labels = labels[dev_end:]

    train_ds = TensorDataset(train_acts, train_labels)
    dev_ds = TensorDataset(dev_acts, dev_labels)
    test_ds = TensorDataset(test_acts, test_labels)

    torch.save(train_ds, os.path.join(acts_save_path, "train.pt"))
    torch.save(train_ds, os.path.join(acts_save_path, "dev.pt"))
    torch.save(train_ds, os.path.join(acts_save_path, "test.pt"))

else:
    train_ds = torch.load(
        os.path.join(acts_save_path, "train.pt"), 
    )
    dev_ds = torch.load(
        os.path.join(acts_save_path, "dev.pt"), 
    )

train_loader = DataLoader(train_ds, batch_size=bs)
dev_loader = DataLoader(dev_ds, batch_size=bs)

if v_end == -1:
    v_end = config.num_hidden_layers

max_seq_len = 123
extra_steps = num_extra_steps * h_step

layers = list(range(v_start, v_end+1, v_step))
token_inds = list(range(h_start-extra_steps, h_end+1, h_step)) \
+ list(range((max_seq_len-1)-extra_steps, max_seq_len, h_step))

for layer in layers:
    for token_ind in token_inds:
        print(f"layer_{layer}_token_{token_ind}")

        probe = LinearProbe(config.hidden_size)
        probe = probe.to(device).to(torch.float32)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss
        optimizer = optim.Adam(probe.parameters(), lr=0.001)

        for epoch in tqdm(range(num_epochs), desc="Training probes"):
            for batch_x, batch_y in tqdm(train_loader):
                batch_x = batch_x[:, layer, token_ind, :].to(device).to(torch.float32)
                batch_y = batch_y.to(device)

                print(batch_x.shape)
                
                outputs = probe(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dev_preds = []
        dev_labels = []
        for batch_x, batch_y in tqdm(dev_loader, 
                                    desc="Testing probes"):
            logits = probe(
                batch_x[:, layer, token_ind, :].to(device).to(torch.float32)
            )
            preds = logits.argmax(dim=-1)
            
            dev_preds.append(preds)
            dev_labels.append(batch_y)

        dev_preds = torch.concatenate(dev_preds).cpu()
        dev_labels = torch.concatenate(dev_labels)
        acc = accuracy_score(dev_preds, dev_labels)

        with open(
            os.path.join(results_save_path, f"layer_{layer}_token_{token_ind}.txt"), 'w'
        ) as fw:
            fw.write(f"Final dev accuracy: {acc:.4f}")