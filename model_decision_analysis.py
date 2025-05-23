# Standard library
import os
import random
import argparse

# Third-party libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Local application imports
from utils import (
    ADMISSIONS_SETTINGS, HIRING_NAMES_SETTINGS,
    HIRING_SETTINGS, ADMISSIONS_NAMES_SETTINGS,
    llm_predict, format_prompt, sample_one
)

# Set seed for all relevant libraries
random_seed = int.from_bytes(os.urandom(4), byteorder="little")
print(f"Using random seed: {random_seed}")

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # For GPU randomness

parser = argparse.ArgumentParser()
parser.add_argument("--task", 
                    choices=["Admissions", "AdmissionsNames", "Hiring", "HiringNames"],
                    default="Admissions"
                    )
parser.add_argument("--template_path", 
                    default="./llm_prediction_bias/prompts/ug_admissions_short.txt"
                    )
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset_size", type=int, default=2000)
parser.add_argument("--model_name", 
                    default='sharpbai/alpaca-7b-merged', 
                    help='Name or path of the model'
                    )
parser.add_argument("--preds_save_path", default='./')
parser.add_argument("--probs_save_path", default='./')
parser.add_argument("--plots_save_path", default='./')

args = parser.parse_args()

task = args.task
template_path = args.template_path
ds_size = args.dataset_size
bs = args.batch_size
model_name = args.model_name

preds_save_path = args.preds_save_path
probs_save_path = args.probs_save_path
plots_save_path = args.plots_save_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(preds_save_path, exist_ok=True)
os.makedirs(probs_save_path, exist_ok=True)
os.makedirs(plots_save_path, exist_ok=True)

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

llama = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
)
_ = llama.to(device)
_ = llama.eval()

if task == 'Admissions':
    settings = ADMISSIONS_SETTINGS
    ds_type = 'admissions_short'
elif task == 'AdmissionsNames':
    settings = ADMISSIONS_NAMES_SETTINGS
    ds_type = 'admissions_names'
elif task == 'Hiring':
    settings = HIRING_SETTINGS
    ds_type = 'hiring_short'
elif task == 'HiringNames':
    settings = HIRING_NAMES_SETTINGS
    ds_type = 'hiring_names'

template = open(template_path).read()
profiles = [sample_one(settings) for _ in range(ds_size)]
prompts = [format_prompt(template, profile, dataset=ds_type) 
            for profile in profiles]

df_data = pd.DataFrame(profiles)
df_data['profile'] = prompts

test_data = Dataset.from_dict({'input_text': df_data['profile']})
test_dataloader = DataLoader(test_data, batch_size=bs)

preds = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Making decisions"):
        output_labels = llm_predict(llama, 
                                    tokenizer, 
                                    # device,
                                    batch['input_text'], 
                                    generate=False, gen_length=3
                                   )
        preds += output_labels

        print(output_labels) # for debugging purposes

preds = ["Yes" if "Yes" in pred else "No" for pred in preds]
_preds = []
for pred in preds:
    if "Yes" in pred:
        _preds.append("Yes")
    elif "No" in pred:
        _preds.append("No")
    else:
        _preds.append("Null")

df_data['pred'] = _preds
df_data = df_data.loc[df_data['pred'] != "Null"]

df_data.to_csv(os.path.join(preds_save_path, 'preds.csv'), index=False)

df_pred_no = df_data.loc[df_data['pred'] == 'No']
df_pred_yes = df_data.loc[df_data['pred'] == 'Yes']

print(f"# Negatives: {len(df_pred_no)}")
print(f"# Positives: {len(df_pred_yes)}")

features = df_data.drop(columns=['profile','pred']).columns
for feature in features:
    print(f"Plotting feature: {feature}")
    
    bin_vals = sorted(df_data[feature].unique().tolist())
    yes_count = df_pred_yes[feature].value_counts()
    no_count = df_pred_no[feature].value_counts()
    
    feat_yes = []
    feat_no = []
    for bin_val in bin_vals:
        if bin_val in yes_count:
            bin_yes_count = yes_count[bin_val]
        else: # zero counts don't show up in the dataframe
            bin_yes_count = 0 
            
        if bin_val in no_count:
            bin_no_count = no_count[bin_val]
        else:
            bin_no_count = 0
            
        total_count = bin_yes_count + bin_no_count
        yes_prob = bin_yes_count / (total_count + 1e-4)
        no_prob = bin_no_count / (total_count + 1e-4)
        
        feat_yes.append(yes_prob)
        feat_no.append(no_prob)

    # saving data
    feature_dict = {
        'values': bin_vals,
        'yes_prob': feat_yes,
        'no_prob': feat_no,
    }
    df_feat = pd.DataFrame(feature_dict).round(decimals=3)
    df_feat.to_csv(
        os.path.join(probs_save_path, f'{feature}.csv'), 
        index=False
    )
    
    # adjusting label angles for labels that are too long
    angle = 0
    if feature in ['letters_quality', 'topic']: # outdated
        angle = 90
    if feature == 'degree':
        angle = 45
    
    # adding a unit for income
    if feature == 'income':
        feature_ = 'income (thousand dollars)'
    else:
        feature_ = feature
    
    x = range(len(bin_vals))
    plt.bar(x, feat_yes, label='Accepted', color='skyblue')
    plt.bar(x, feat_no, bottom=feat_yes, label='Rejected', color='salmon')

    plt.xlabel(feature_, fontsize=16)
    plt.ylabel('Decision Probability', fontsize=16)
    plt.xticks(x, bin_vals, rotation=angle) 
    plt.legend()
    plt.savefig(
        os.path.join(plots_save_path, f'llama_pred_feature_{feature}.jpg'), bbox_inches='tight'
    )
    plt.close()