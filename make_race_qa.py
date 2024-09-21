import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from torch.utils.data import DataLoader

from make_ctf_dataset import HIRE_DEC_NAMES_SETTINGS
from eval_alignment import load_alignment

import sys
sys.path.append('../pyvene/')
import pyvene as pv

# def classify_race(prompt):
#     race_keywords = {
#         'White': ['white', 'caucasian'],
#         'Black': ['black', 'african'],
#         'Asian': ['asian', 'japanese'],
#         'Latino': ['hispanic', 'latin'],
#     }
#     prompt_words = prompt.lower()
#     for race, kws in race_keywords.items():
#         for kw in kws:
#             if kw in prompt_words:
#                 return race
#     return "undefined"


race_qa_prompt = """You are assisting a hiring manager in reviewing job applications. [NAME] has a college degree and 5 years of experience. They have good letters of recommendations. What is the applicant's race? If the applicant is Black, say 'Black'. If they are Asian, say 'Asian'. If they are White, say 'White'. If they are Latino, say 'Latino'. Do not say anything other than these.

Answer:
"""

race_qa_prompt_gemma = """You are assisting a hiring manager in reviewing job applications. [NAME] has a college degree and 5 years of experience. They have good letters of recommendations. What is the applicant's race? You can infer this from their name. Feel free to make a guess if you are not sure. If you think the applicant is Black, say 'Black'. If Asian, say 'Asian'. If White, say 'White'. If Latino, say 'Latino'. Do not say anything other than these and do not decline my request. For example:

Name: Brady
Answer: White

Name: Jamal
Answer: Black

Name: Jose
Answer: Latino

Name: Eri
Answer: Asian

Answer:
"""

# device = 'cuda:1'

# Making a simple "RaceQA" dataset
names = HIRE_DEC_NAMES_SETTINGS['name']
prompts = []
races = []

for race, name_list in names.items():
    for name in name_list:
        prompt = race_qa_prompt_gemma.replace("[NAME]", name)
        prompts.append(prompt)
        races.append(race)
        
df_qa = pd.DataFrame({'race': races, 'profile': prompts})
df_qa['pred'] = df_qa['race']
df_qa.to_csv('./datasets/race_qa_gemma.csv', index=False)



# qa_dataset = Dataset.from_pandas(df_qa)
# qa_loader = DataLoader(qa_dataset, batch_size=64)

# bs = 64
# collect_layer = 2
# collect_pos = 17

# admissions_path = './llm_prediction_bias/datasets/admissions'

# df_admissions_race = pd.read_csv(os.path.join(admissions_path, 'pred.csv'))

# races = df_admissions_race['race'].unique().tolist()

# race_datasets = [Dataset.from_pandas(
#     df_admissions_race.loc[df_admissions_race['race'] == race]) for race in races
# ]
# race_loaders = [DataLoader(race_ds, batch_size=bs) for race_ds in race_datasets]

# config_collect = pv.IntervenableConfig([
#     {
#         "layer": collect_layer,
#         "component": "block_output",
#         "intervention_type": pv.CollectIntervention
#     }
# ])
# vene_collect = pv.IntervenableModel(config_collect, llama)
# vene_collect.set_device(device)
# vene_collect.disable_model_gradients()

# align_path = './llm_prediction_bias/alignments/synthetic_short_high_iia/layer_2_pos_17'

# bdas_config = pv.IntervenableConfig(
#     [{
#         "layer": layer,
#         "component": "block_output",
#         "intervention_type": pv.BoundlessRotatedSpaceIntervention,
#     }
#         for layer in range(2, 3)
#     ]
# )

# vene_bdas = load_alignment(align_path, bdas_config, llama)
# vene_bdas.set_device(device)
# vene_bdas.disable_model_gradients()

# races_activations = []
# for race_loader in race_loaders:
    
#     race_activations = []
#     for example in tqdm(race_loader):
#         base_tokens = tokenizer(example['base'], 
#                                 return_tensors='pt', 
#                                 padding=True).to(device)
#         vene_out = vene_collect(
#             base_tokens,                                 
#             unit_locations={'base': collect_pos}
#         )
#         activations = vene_out[0][1]
#         activations = torch.concatenate(activations)
#         race_activations.append(activations)
        
#     race_activations_pt = torch.concatenate(race_activations)
#     races_activations.append(race_activations_pt)


# dist = 15 # distance to mention of the name
# # patch = 3 # expected number of tokens in a name
# patch = 4
# patches = np.arange(dist, dist + patch).tolist()
# patches


# for i in range(len(race_mean_acts)):

#     all_base_gens = []
#     all_ctf_gens = []

#     with torch.no_grad():
#         for example in tqdm(qa_loader):
#             batch = example['prompt']
#             base_tokens = tokenizer(batch, 
#                                     return_tensors='pt', 
#                                     padding=True).to(device)

#             num_pos = len(patches)
#             src_activations = (src_act.reshape(1, 1, -1)
#                             .expand(len(batch), num_pos, -1) # shape (bs, num_pos, hidden)
#                             )

#             base_outputs, ctf_outputs = vene_bdas.generate(
#                 base_tokens,
#                 source_representations = src_activations,
#                 max_length = base_tokens.input_ids.shape[1] + 20,
#                 output_original_output = True,
#                 intervene_on_prompt = True,
#                 unit_locations = {"base": patches}
#             )

#             base_gen = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
#             ctf_gen = tokenizer.batch_decode(ctf_outputs, skip_special_tokens=True)

#             all_base_gens += base_gen
#             all_ctf_gens += ctf_gen

#     # _all_base_gens = []
#     # _all_ctf_gens = []

#     # for j in range(len(df_qa)):
#     #     prompt = df_qa.iloc[j]['prompt']
#     #     base_gen = all_base_gens[j]
#     #     _all_base_gens.append(base_gen[len(prompt):])

#     # for j in range(len(df_qa)):
#     #     prompt = df_qa.iloc[j]['prompt']
#     #     ctf_gen = all_ctf_gens[j]
#     #     _all_ctf_gens.append(ctf_gen[len(prompt):])

#     df_qa['base_race'] = all_base_gens
#     df_qa['ctf_race'] = all_ctf_gens


#     # base_races = [classify_race(prompt) for prompt in _all_base_gens]
#     # ctf_races = [classify_race(prompt) for prompt in _all_ctf_gens]

#     # df_qa['base_race'] = base_races
#     # df_qa['ctf_race'] = ctf_races

#     # base_race_counts = df_qa['base_race'].value_counts().reset_index().rename(
#     #     columns={'base_race': 'race', 'count': 'base_count'}
#     # )
#     # ctf_race_counts = df_qa['ctf_race'].value_counts().reset_index().rename(
#     #     columns={'ctf_race': 'race', 'count': 'ctf_count'}
#     # )

#     # race_counts = base_race_counts.merge(ctf_race_counts, on='race', how='left').fillna(0)
#     # race_counts = race_counts.set_index('race').loc[races]
#     # lst_race_counts.append(race_counts)