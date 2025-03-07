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
# sns.set_context("notebook")

config = AutoConfig.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset('csv', data_files={
    'train': os.path.join(ds_path, 'train.csv'),
    # 'test': os.path.join(ds_path, 'test.csv'),
})

if v_end == -1:
    v_end = config.num_hidden_layers

token_ids = tokenizer(ds['train']['base'][:50], 
                      padding=True, 
                      return_tensors="pt").input_ids
max_seq_len = token_ids.shape[1]
extra_steps = num_extra_steps * h_step

# breakpoint()

layers = list(range(v_start, v_end+1, v_step))
positions = list(range(h_start-extra_steps, h_end+1, h_step))

plot_end = True # whether to plot tokens near the end
if plot_end:
    positions += list(range(-1-extra_steps, 0, h_step))

res_matrix = np.zeros((len(layers), len(positions)))

for i in range(len(layers)):
    layer = layers[i]
    for j in range(len(positions)):
        position = positions[j]
        filename = f'layer_{layer}_token_{position}.txt'
        
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

tokens = tokenizer.batch_decode(token_ids[0])
tokens_search = tokens[h_start-extra_steps : h_end+1 : h_step] \
+ tokens[(max_seq_len-1)-extra_steps : max_seq_len : h_step]

# breakpoint()

x_labels = []
for token, pos in zip(tokens_search, positions):
    if pos == -93 or pos == -92:
        token = '<name>'
    if token.strip() in ['Asian', 'Black', 'Latino', 'White']:
        token = '<race>'
    if '\n' in token:
        token = token.replace('\n', '\\n')
    x_labels.append(f'{token} ({pos})')

# Plotting
plt.figure(figsize=(13, 9))
# plt.figure(figsize=(13, 12)) # for when there are many layers
# plt.figure(figsize=(13, 14))
sns.heatmap(np.flip(res_matrix, axis=0), 
            annot=True, 
            # annot_kws={'size':20}, 
            annot_kws={'size':18}, 
            fmt=".2f", cmap="magma_r", cbar=False, 
            xticklabels=x_labels, yticklabels=layers_r)

plt.xticks(fontsize=25, rotation=45, ha='right')
plt.yticks(fontsize=25)
plt.ylabel("Layer", fontsize=30)

plt.savefig(save_file + ".jpg", bbox_inches='tight')
plt.savefig(save_file + ".pdf", bbox_inches='tight')
