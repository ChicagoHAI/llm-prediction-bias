import argparse
import os
import pandas as pd
from utils import format_label

"""
Subsamples the DataFrame to balance the number of examples for each 
counterfactual behavior.

Parameters:
df (pd.DataFrame): The input DataFrame containing the data to be balanced.

Returns:
pd.DataFrame: The balanced DataFrame.
"""
def balance_labels(df):
    labels_count = df[['base_label', 'src_label']].value_counts().tolist()
    n_samples = min(labels_count) if labels_count else 0

    labels = df['base_label'].unique()
    df_ctfs = []
    for base_label in labels:
        for src_label in labels:
            df_base_src = df.loc[
                (df['base_label'] == base_label) & 
                (df['src_label'] == src_label)
            ]
            if len(df_base_src) >= n_samples and n_samples > 0:
                df_base_src = df_base_src.sample(n_samples)
            else:
                df_base_src = pd.DataFrame([])
            df_ctfs.append(df_base_src)

    df_ctf_balanced = pd.concat(df_ctfs, axis=0) \
    .sample(frac=1).reset_index(drop=True)

    return df_ctf_balanced


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", help="path to the base predictions")
    parser.add_argument("--source_path", help="path to the source predictions")

    parser.add_argument("--model_name", choices=['alpaca', 'llama3', 'mistral', 'gemma'])
    parser.add_argument("--causal_variable")
    parser.add_argument("--side_variables", nargs='*', type=str, 
                        help="input variables that do not affect the causal variable")
    
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_dev_split", nargs='+', type=float)
    parser.add_argument("--save_path", default='./')

    args = parser.parse_args()

    base_path = args.base_path
    src_path = args.source_path

    model_name = args.model_name
    causal_var = args.causal_variable
    side_vars = args.side_variables

    seed = args.random_seed
    train_dev_split = args.train_dev_split
    save_path = args.save_path

    df_base = pd.read_csv(base_path)
    df_src = pd.read_csv(src_path)
    df_base['var'] = df_base[causal_var]
    df_src['var'] = df_src[causal_var]
    
    var_values = df_base['var'].unique()
    labels = df_base['pred'].unique()

    all_df_ctf = []
    for value in var_values:
        for base_label in labels:
            for src_label in labels:
                print(f"{value}: {base_label} -> {src_label}")

                counterfactual_settings = df_base.loc[
                    (df_base['var'] == value) 
                    & (df_base['pred'] == src_label)
                ]
                side_settings = df_base.merge(counterfactual_settings[side_vars], on=side_vars)
                
                if len(side_settings) > 0:
                    df_ctf = pd.DataFrame([])

                    # sampling base examples
                    base_settings = side_settings.loc[
                        (side_settings['pred'] == base_label)
                    ]
                    df_ctf['base'] = base_settings['profile'].reset_index(drop=True)
                    
                    # sampling source examples
                    df_race = df_src.loc[df_src['var'] == value] \
                    .sample(len(df_ctf), replace=True, random_state=seed) \
                    .reset_index(drop=True)

                    df_ctf['source'] = df_race['profile']
                    df_ctf['base_label'] = base_label
                    df_ctf['src_label'] = src_label

                    df_ctf = pd.concat(
                        [df_ctf, base_settings[side_vars].reset_index(drop=True)],
                        axis=1
                    )
                    all_df_ctf.append(df_ctf)
    all_df_ctf = pd.concat(all_df_ctf, axis=0)

    # doing manual tokenization because the model's output Yes or No tokens
    # can be different from the tokenizer's, weird.
    all_df_ctf[['base_label','src_label']] = all_df_ctf[['base_label','src_label']] \
    .replace('Yes', format_label('Yes', model_name)) \
    .replace('No', format_label('No', model_name))
    all_df_ctf = all_df_ctf.dropna()

    len_df = len(all_df_ctf)
    train_frac, dev_frac = train_dev_split
    train_end = int(round(train_frac * len_df))
    dev_end = int(
        round((train_frac + dev_frac) * len_df)
    )

    all_df_ctf = all_df_ctf.sample(frac=1).reset_index(drop=True)
    df_train = all_df_ctf[:train_end]
    df_dev = all_df_ctf[train_end:dev_end]
    df_test = all_df_ctf[dev_end:]

    if 'uni' in side_vars:
        group = 'uni'
    elif 'role' in side_vars:
        group = 'role'
    df_train_balanced = df_train.groupby(group).apply(balance_labels)
    df_dev_balanced = df_dev.groupby(group).apply(balance_labels)
    df_test_balanced = df_test.groupby(group).apply(balance_labels)

    train_ctf_counts = df_train_balanced[['base_label','src_label']].value_counts()
    print(train_ctf_counts)
    dev_ctf_counts = df_dev_balanced[['base_label','src_label']].value_counts()
    print(dev_ctf_counts)
    test_ctf_counts = df_test_balanced[['base_label','src_label']].value_counts()
    print(test_ctf_counts)

    os.makedirs(save_path, exist_ok=True)
    df_train_balanced.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_dev_balanced.to_csv(os.path.join(save_path, 'dev.csv'), index=False)
    df_test_balanced.to_csv(os.path.join(save_path, 'test.csv'), index=False)

if __name__ == "__main__":
    main()