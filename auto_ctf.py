import pandas as pd
import argparse
import os

def race_to_race(row):
    return row['race']

def name_to_race(name):
    pass

parser = argparse.ArgumentParser()

parser.add_argument("--base_path")
parser.add_argument("--source_path")
parser.add_argument("--causal_variable", choices=['race', 'race_given_name'])
# parser.add_argument("--structural_func", 
#                     help="the structural function for var", 
#                     choices=['race->race', 'name->race'])
parser.add_argument("--p_variables", nargs='+', type=str, 
                    help="input variables that affect var")
parser.add_argument("--q_variables", nargs='+', type=str, 
                    help="input variables that do not affect var")
parser.add_argument("--save_path", default='./')

args = parser.parse_args()

base_path = args.base_path
src_path = args.source_path
causal_var = args.causal_variable
p_vars = args.p_variables # should be defined in the causal function already
q_vars = args.q_variables
save_path = args.save_path

os.makedirs(save_path, exist_ok=True)

if causal_var == 'race':
    var_name = 'race'
    var_func = race_to_race
elif causal_var == 'race_given_name':
    var_name = 'race'
    var_func = name_to_race

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
                df = df_base.loc[
                    (df_base[q_vars] == row[q_vars]).all(axis=1) & 
                    (df_base['var'] != value)
                ]
                dfs.append(df)
                
            if len(dfs) > 0:
                q_settings = pd.concat(dfs, axis=0)
                q_base_settings = q_settings.loc[q_settings['pred'] == base_label]
                
                df_ctf = pd.DataFrame([])

                # sampling base examples
                df_ctf['base'] = q_base_settings['profile'].reset_index(drop=True)
                
                # sampling source examples
                df_race = df_src.loc[df_src['var'] == value].sample(len(df_ctf), replace=True).reset_index(drop=True)
                df_ctf['source'] = df_race['profile']
                
                df_ctf['base_label'] = base_label
                df_ctf['src_label'] = src_label
                
                all_df_ctf.append(df_ctf)

all_df_ctf = pd.concat(all_df_ctf, axis=0)

ctf_behavior_counts = []
for base_label in labels:
    for src_label in labels:
        base_src = all_df_ctf.loc[
            (all_df_ctf['base_label'] == base_label) & 
            (all_df_ctf['src_label'] == src_label)
        ]
        ctf_behavior_counts.append(len(base_src))

n_samples = min(ctf_behavior_counts)

df_ctfs = []
for base_label in labels:
    for src_label in labels:
        df_base_src = all_df_ctf.loc[
            (all_df_ctf['base_label'] == base_label) & 
            (all_df_ctf['src_label'] == src_label)
        ].sample(n_samples)
        df_ctfs.append(df_base_src)

df_ctf_balanced = pd.concat(df_ctfs, axis=0)

df_ctf_balanced[['base_label','src_label']] = df_ctf_balanced[['base_label','src_label']].replace('Yes', 8241).replace('No', 3782)

df_ctf_balanced.to_csv(os.path.join(save_path, 'test.csv'), index=False)