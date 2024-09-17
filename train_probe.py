
import os
from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../pyvene/')
import pyvene as pv


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        return probs

def race_to_label(race):
    if race == 'White':
        return 0
    elif race == 'Asian':
        return 1
    elif race == 'Black':
        return 2
    elif race == 'Latino':
        return 3

get_race = lambda loc, prompt: prompt.split(' ')[loc]

def process_dataset(dataset):
    dataset = dataset.map(lambda x: {'race': get_race(12, x['base'])})
    dataset = dataset.map(lambda x: {'label': race_to_label(x['race'])})
    dataset = dataset.select_columns(['base','label'])
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input arguments for cross-task intervention on HireDec.")

    parser.add_argument("--model_name", type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')
    parser.add_argument("--dataset_path", type=str, 
                        default='./datasets/admissions_short_race')
    parser.add_argument("--save_path", type=str, 
                        default='./')

    parser.add_argument("--train_size", type=int, default=500)
    
    parser.add_argument("--horizontal_start", type=int, default=0)
    parser.add_argument("--horizontal_end", type=int, default=50)
    parser.add_argument("--horizontal_step", type=int, default=1)

    parser.add_argument("--vertical_start", type=int, default=0)
    parser.add_argument("--vertical_end", type=int, default=-1)
    parser.add_argument("--vertical_step", type=int, default=1)

    parser.add_argument("--extra_steps", 
                        help="""The number of steps before {h_pos} to search.""", 
                        default=4, type=int)

    args = parser.parse_args()
    return args

args = parse_args()

model_name = args.model_name
ds_path = args.dataset_path
save_path = args.save_path
train_size = args.train_size

h_start = args.horizontal_start
h_end = args.horizontal_end
h_step = args.horizontal_step
num_extra_steps = args.extra_steps

v_start = args.vertical_start
v_end = args.vertical_end
v_step = args.vertical_step

bs = 64

os.makedirs(save_path, exist_ok=True)

device = 'cuda:1'

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


# df_race = pd.read_csv(
#     os.path.join(ds_path, 'train.csv')
# )[['base']].sample(1000)
# df_race['race'] = df_race['base'].apply(partial(get_race, 12))
# df_race['label'] = df_race['race'].apply(race_to_label)

# ds_race = Dataset.from_pandas(df_race[['base', 'label']])
# ds_loader = DataLoader(ds_race, batch_size=64)

ds = load_dataset('csv', data_files={
    'train': os.path.join(ds_path, 'train.csv'),
    'dev': os.path.join(ds_path, 'dev.csv'),
    'test': os.path.join(ds_path, 'test.csv'),
})

ds_train = ds['train'].shuffle(seed=42).select(range(train_size))
ds_train = process_dataset(ds_train)
train_loader = DataLoader(ds_train, batch_size=bs)

ds_dev = process_dataset(ds['dev'])
dev_loader = DataLoader(ds_dev, batch_size=bs)

# if h_end == -1:
#     h_end = len(tokenizer(ds_race[0]['base']).input_ids)

if v_end == -1:
    v_end = llama.config.num_hidden_layers

max_seq_len = len(tokenizer(ds_train[0]['base']).input_ids)
extra_steps = num_extra_steps * h_step

layers = list(range(v_start, v_end+1, v_step))
token_inds = list(range(h_start-extra_steps, h_end+1, h_step)) \
+ list(range((max_seq_len-1)-extra_steps, max_seq_len, h_step))

for layer in layers:
    config_collect = pv.IntervenableConfig([
        {
            "layer": layer,
            "component": "block_output",
            "intervention_type": pv.CollectIntervention,
        }
    ])
    vene_collect = pv.IntervenableModel(config_collect, llama)
    vene_collect.set_device(device)
    vene_collect.disable_model_gradients()

    for token_ind in token_inds:
        print(f"layer_{layer}_token_{token_ind}")

        print(token_ind)

        # collecting train activations
        train_acts = []
        for batch in tqdm(train_loader, desc="Getting train activations"):
            batch_toks = tokenizer(batch['base'], 
                                return_tensors="pt", 
                                padding=True).to(device)
            vene_out = vene_collect(
                batch_toks,                                 
                unit_locations={'base': token_ind}
            )
            activations = vene_out[0][1]
            activations = torch.concatenate(activations)
            train_acts.append(activations)
        train_acts = torch.concatenate(train_acts)

        # collecting dev activations
        dev_acts = []
        for batch in tqdm(dev_loader, desc="Getting dev activations"):
            batch_toks = tokenizer(batch['base'], 
                                return_tensors="pt", 
                                padding=True).to(device)
            vene_out = vene_collect(
                batch_toks,                                 
                unit_locations={'base': token_ind}
            )
            activations = vene_out[0][1]
            activations = torch.concatenate(activations)
            dev_acts.append(activations)
        dev_acts = torch.concatenate(dev_acts)

        # labels_pt = torch.tensor(ds_race['label'])
        # train_ds = TensorDataset(race_acts_pt[:n_train], labels_pt[:n_train])
        # test_ds = TensorDataset(race_acts_pt[n_train:], labels_pt[n_train:])

        # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        # test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

        ds_train_acts = TensorDataset(
            train_acts, torch.tensor(ds_train['label'])
        )
        ds_dev_acts = TensorDataset(
            dev_acts, torch.tensor(ds_dev['label'])
        )

        train_acts_loader = DataLoader(ds_train_acts, batch_size=bs)
        dev_acts_loader = DataLoader(ds_dev_acts, batch_size=bs)

        # Initialize the linear probe
        input_dim = llama.config.hidden_size
        probe = LinearProbe(input_dim)
        probe = probe.to(device).to(torch.float32)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss
        optimizer = optim.Adam(probe.parameters(), lr=0.001)

        # Training loop
        num_epochs = 20
        for epoch in tqdm(range(num_epochs), desc="Training the probe"):
            for batch_x, batch_y in tqdm(train_acts_loader):
                batch_x = batch_x.to(device).to(torch.float32)
                batch_y = batch_y.to(device)
                
                outputs = probe(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dev_preds = []
        dev_labels = []
        for batch_x, batch_y in tqdm(dev_acts_loader, 
                                     desc="Testing the probe"):
            logits = probe(batch_x.to(device).to(torch.float32))
            preds = logits.argmax(dim=-1)
            
            dev_preds.append(preds)
            dev_labels.append(batch_y)

        dev_preds = torch.concatenate(dev_preds).cpu().numpy()
        dev_labels = np.concatenate(dev_labels)
        acc = accuracy_score(dev_preds, dev_labels)

        with open(
            os.path.join(save_path, f"layer_{layer}_token_{token_ind}.txt"), 'w'
        ) as fw:
            fw.write(f"Final dev accuracy: {acc:.4f}")
