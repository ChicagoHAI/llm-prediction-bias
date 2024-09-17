import pandas as pd
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import argparse
from matplotlib import pyplot as plt
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../pyvene/')
import pyvene as pv # using local pyvene

from utils import llm_predict
from make_ctf_dataset import HIRE_DEC_NAMES_SETTINGS, format_prompt, sample_one, \
HIRE_DEC_EVAL_SETTINGS, HIRING_SETTINGS_SHORT, BIOS_SETTINGS_SHORT


def get_race_pos(prompt):
    tokens = tokenizer(prompt.lower()).input_ids
    for i in range(len(tokens)):
        word = tokenizer.decode(tokens[i])
        if word in ['white', 'black', 'pan', 'ian']:
            return i
        
def get_length(prompt):
    tokens = tokenizer(prompt)
    return len(tokens.input_ids)


parser = argparse.ArgumentParser()

parser.add_argument("--task", 
                    choices=["Admissions", "HireDec", "HireDecEval", "HireDecNames", "DiscrimEval"],
                    default="Admissions"
                    )
parser.add_argument("--template_path", 
                    default="./llm_prediction_bias/prompts/ug_admissions_short.txt"
                    )
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
model_name = args.model_name

preds_save_path = args.preds_save_path
probs_save_path = args.probs_save_path
plots_save_path = args.plots_save_path

device = 'cuda'

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
    torch_dtype=torch.bfloat16,  # save memory
)
_ = llama.to(device)
_ = llama.eval()

if task == 'DiscrimEval':
    dataset = load_dataset('Anthropic/discrim-eval', 'explicit')
    dataset = dataset.rename_column('filled_template', 'profile')
    dataset = dataset.remove_columns(["decision_question_id", "fill_type"])
    # dataset = dataset['train'].shuffle(seed=42).select(range(ds_size))

    df_data = dataset['train'].to_pandas()

    df_data['race_idx'] = df_data['profile'].apply(get_race_pos)
    df_data['prompt_len'] = df_data['profile'].apply(get_length)

    df_data = df_data.loc[df_data['race'] != 'Native American']
    df_data['race'] = df_data['race'].replace({'white': 'White', 'Hispanic': 'Latino'}) 

    # selecting prompts with similar race positions
    df_data = df_data.loc[
        (df_data['race_idx'] > 15) & (df_data['race_idx'] < 20) & 
        (df_data['prompt_len'] > 140) & (df_data['prompt_len'] < 150)
    ]
    print(len(df_data))

else:
    if task == 'Admissions':
        settings = BIOS_SETTINGS_SHORT
        settings['uni'] = ['Harvard']
        ds_type = 'admissions_short'
    elif task == 'HireDec':
        settings = HIRING_SETTINGS_SHORT
        ds_type = 'hiring_short'
    elif task == 'HireDecEval':
        settings = HIRE_DEC_EVAL_SETTINGS
        ds_type = 'hire_dec_eval'
    elif task == 'HireDecNames':
        settings = HIRE_DEC_NAMES_SETTINGS
        ds_type = 'hire_dec_names'

    template = open(template_path).read()

    profiles = [sample_one(settings) for _ in range(ds_size)]
    prompts = [format_prompt(template, profile, dataset=ds_type) for profile in profiles]

    df_data = pd.DataFrame(profiles)
    df_data['profile'] = prompts

test_data = Dataset.from_dict({'input_text': df_data['profile']})
test_dataloader = DataLoader(test_data, batch_size=64)

preds = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        output_labels = llm_predict(llama, 
                                    tokenizer, 
                                    device,
                                    batch['input_text'], 
                                    generate=True, gen_length=3
                                   )
        preds += output_labels

preds = ["Yes" if "Yes" in pred else "No" for pred in preds]
df_data['pred'] = preds

df_data.to_csv(os.path.join(preds_save_path, 'preds.csv'), index=False)

df_pred_no = df_data.loc[df_data['pred'] == 'No']
df_pred_yes = df_data.loc[df_data['pred'] == 'Yes']

# features = list(settings.keys())
features = df_data.drop(columns=['profile','pred']).columns
for feature in features:
    print(feature)
    
    # bin_vals = settings[feature]
    bin_vals = df_data[feature].unique().tolist()
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
        yes_prob = bin_yes_count / total_count
        no_prob = bin_no_count / total_count
        
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
