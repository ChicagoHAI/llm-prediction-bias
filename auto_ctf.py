import pandas as pd
import argparse
import os
import re

from utils import format_label

def race_to_race(row):
    return row['race']

def get_role(prompt):
    match = re.search(r'for a\s*(.*?)\s*role', prompt)
    if match:
        return match.group(1)

def split_by_role_and_save_csv(df, save_path, data_split='test'):
    df['role'] = df['base'].apply(get_role)
    df['role'] = df['role'].apply(lambda x: x.lower().replace(" ", "-"))
    roles = df['role'].unique()

    for role in roles:
        save_path_role = save_path.format(role)
        os.makedirs(save_path_role, exist_ok=True)
        
        df_role = df.loc[df['role'] == role]
        print(f"role: {role}, size: {len(df_role)}")
        df_role.to_csv(
            os.path.join(save_path_role, f"{data_split}.csv"), index=False
        )

parser = argparse.ArgumentParser()

parser.add_argument("--base_path")
parser.add_argument("--source_path")
parser.add_argument("--model_name", choices=['alpaca', 'llama3', 'mistral', 'gemma'])
parser.add_argument("--causal_variable", choices=['race', 'race_given_name'])
parser.add_argument("--p_variables", nargs='+', type=str, 
                    help="input variables that affect var")
parser.add_argument("--q_variables", nargs='*', type=str, 
                    help="input variables that do not affect var")
parser.add_argument("--train_dev_split", nargs='+', type=float)
parser.add_argument("--save_path", default='./')
parser.add_argument("--save_by_role", action='store_true')

args = parser.parse_args()

base_path = args.base_path
src_path = args.source_path
model_name = args.model_name

causal_var = args.causal_variable
p_vars = args.p_variables # should be defined in the causal function already
q_vars = args.q_variables
train_dev_split = args.train_dev_split
save_path = args.save_path
save_by_role = args.save_by_role

if causal_var == 'race':
    var_name = 'race'
    var_func = race_to_race
elif causal_var == 'race_given_name':
    pass

df_base = pd.read_csv(base_path)
df_src = pd.read_csv(src_path)

df_base['var'] = df_base.apply(var_func, axis=1)
df_src['var'] = df_src.apply(var_func, axis=1)

var_values = df_base['var'].unique()
labels = df_base['pred'].unique()

all_df_ctf = []
for value in var_values:
    for base_label in labels:
        for src_label in labels:
            print(f"{value} {base_label} {src_label}")
            
            p_q_base_settings = df_base.loc[(df_base['var'] == value) &
                                            (df_base['pred'] == src_label)]
            
            dfs = []
            for _, row in p_q_base_settings.iterrows():
                if len(q_vars) > 0:
                    df = df_base.loc[(df_base[q_vars] == row[q_vars]).all(axis=1)]
                else:
                    df = df_base
                dfs.append(df)
                
            if len(dfs) > 0:
                q_settings = pd.concat(dfs, axis=0)
                df_ctf = pd.DataFrame([])

                # sampling base examples
                q_base_settings = q_settings.loc[q_settings['pred'] == base_label]
                df_ctf['base'] = q_base_settings['profile'].reset_index(drop=True)
                
                # sampling source examples
                df_race = df_src.loc[df_src['var'] == value] \
                .sample(len(df_ctf), replace=True, random_state=42) \
                .reset_index(drop=True)

                df_ctf['source'] = df_race['profile']
                df_ctf['base_label'] = base_label
                df_ctf['src_label'] = src_label

                # breakpoint()
                df_ctf = pd.concat(
                    [df_ctf, q_base_settings[q_vars].reset_index(drop=True)],
                    axis=1
                )

                all_df_ctf.append(df_ctf)

all_df_ctf = pd.concat(all_df_ctf, axis=0)

ctf_behavior_counts = []
for base_label in labels:
    for src_label in labels:
        base_src = all_df_ctf.loc[
            (all_df_ctf['base_label'] == base_label) & 
            (all_df_ctf['src_label'] == src_label)
        ]
        print(f"{base_label} {src_label}: {len(base_src)}")
        ctf_behavior_counts.append(len(base_src))

nonzero_counts = [count for count in ctf_behavior_counts if count != 0]
n_samples = min(nonzero_counts)

df_ctfs = []
for base_label in labels:
    for src_label in labels:
        df_base_src = all_df_ctf.loc[
            (all_df_ctf['base_label'] == base_label) & 
            (all_df_ctf['src_label'] == src_label)
        ].sample(n_samples)
        df_ctfs.append(df_base_src)

df_ctf_balanced = pd.concat(df_ctfs, axis=0) \
.sample(frac=1).reset_index(drop=True)

# doing manual tokenization because the model's output Yes or No tokens
# can be different from the tokenizer's, weird.
df_ctf_balanced[['base_label','src_label']] = df_ctf_balanced[['base_label','src_label']] \
.replace('Yes', format_label('Yes', model_name)) \
.replace('No', format_label('No', model_name))

df_ctf_balanced = df_ctf_balanced.dropna()

len_df = len(df_ctf_balanced)
train_frac, dev_frac = train_dev_split

train_end = int(round(train_frac * len_df))
dev_end = int(
    round((train_frac + dev_frac) * len_df)
)

df_train = df_ctf_balanced[:train_end]
df_dev = df_ctf_balanced[train_end:dev_end]
df_test = df_ctf_balanced[dev_end:]

if save_by_role: # only for hiring
    split_by_role_and_save_csv(df_train, save_path, 'train')
    split_by_role_and_save_csv(df_dev, save_path, 'dev')
    split_by_role_and_save_csv(df_test, save_path, 'test')
else:
    os.makedirs(save_path, exist_ok=True)
    df_train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_dev.to_csv(os.path.join(save_path, 'dev.csv'), index=False)
    df_test.to_csv(os.path.join(save_path, 'test.csv'), index=False)