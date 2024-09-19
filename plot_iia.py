import numpy as np
import argparse
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import LlamaTokenizer, LlamaConfig, \
AutoConfig, AutoTokenizer


parser = argparse.ArgumentParser()

parser.add_argument("--results_path", 
                    help="Path to the directory containing training results.")
parser.add_argument("--dataset_path", help="""Path the the directory containing
                    the dataset files.""")
parser.add_argument("--model_name", type=str, 
                    default='sharpbai/alpaca-7b-merged', 
                    help='Name or path of the model')

# Training args
parser.add_argument("--horizontal_start", type=int, default=0)
parser.add_argument("--horizontal_end", type=int, default=50)
parser.add_argument("--horizontal_step", type=int, default=1)

parser.add_argument("--vertical_start", type=int, default=0)
parser.add_argument("--vertical_end", type=int, default=-1)
parser.add_argument("--vertical_step", type=int, default=1)

parser.add_argument("--extra_steps", 
                    help="""The number of steps before {h_pos} to search.""", 
                    default=4, type=int)

parser.add_argument("--save_file", help="Path to save the resulting plot.")

args = parser.parse_args()

name = args.model_name
results_path = args.results_path
ds_path = args.dataset_path
save_file = args.save_file

h_start = args.horizontal_start
h_end = args.horizontal_end
h_step = args.horizontal_step
num_extra_steps = args.extra_steps

v_start = args.vertical_start
v_end = args.vertical_end
v_step = args.vertical_step

os.makedirs(os.path.dirname(save_file), exist_ok=True)

# name = "sharpbai/alpaca-7b-merged"
# config = LlamaConfig.from_pretrained(name)
# tokenizer = LlamaTokenizer.from_pretrained(name)

config = AutoConfig.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)

ds = load_dataset('csv', data_files={
    'train': os.path.join(ds_path, 'train.csv'),
    'test': os.path.join(ds_path, 'test.csv'),
})
train_loader = DataLoader(ds['train'], batch_size=32)

if v_end == -1:
    v_end = config.num_hidden_layers

token_ids = tokenizer(ds['train'][0]['base']).input_ids
max_seq_len = len(token_ids)
# max_seq_len = 124
print(max_seq_len)
extra_steps = num_extra_steps * h_step

layers = list(range(v_start, v_end+1, v_step))
positions = list(range(h_start-extra_steps, h_end+1, h_step))
# + list(range((max_seq_len-1)-extra_steps, max_seq_len, h_step))

plot_end = True # whether to plot tokens near the end
if plot_end:
    positions += list(
        range((max_seq_len-1)-extra_steps, max_seq_len, h_step)
    )

res_matrix = np.zeros((len(layers), len(positions)))

for i in range(len(layers)):
    layer = layers[i]
    for j in range(len(positions)):
        position = positions[j]
        filename = f'layer_{layer}_pos_{position}.txt'
        
        try:
            with open(os.path.join(results_path, filename), 'r') as fr:
                line = fr.readline()
            acc = float(line.split(': ')[1])
        except Exception as e:
            print(e)
            acc = 0
            
        res_matrix[i, j] = acc

layers_r = list(layers)
layers_r.reverse()

tokens = tokenizer.batch_decode(token_ids)
# print(len(tokens))
tokens_search = tokens[h_start-extra_steps : h_end+1 : h_step] \
+ tokens[(max_seq_len-1)-extra_steps : max_seq_len : h_step]

x_labels = []
for token, pos in zip(tokens_search, positions):
    x_labels.append(f'{token} ({pos})')

# Plotting
plt.figure(figsize=(10, 5))
sns.heatmap(np.flip(res_matrix, axis=0), 
            annot=True, annot_kws={'size':12}, fmt=".2f", cmap="magma_r", cbar=False, 
            xticklabels=x_labels, yticklabels=layers_r)

plt.title("IIA", fontsize=20)
plt.xticks(fontsize=14, rotation=45, ha='right')
plt.yticks(fontsize=14)
plt.xlabel("Token position", fontsize=14)
plt.ylabel("Layer", fontsize=14)

plt.savefig(save_file, bbox_inches='tight')

