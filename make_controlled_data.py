# Standard library
import argparse
import os
import random

# Third-party libraries
import numpy as np
import pandas as pd

# Local application imports
from utils import (
    ADMISSIONS_NAMES_SETTINGS,
    ADMISSIONS_SETTINGS,
    HIRING_NAMES_SETTINGS,
    HIRING_SETTINGS,
    format_prompt,
    sample_one,
)

parser = argparse.ArgumentParser(description="Generate controlled data for debiasing.")
parser.add_argument('--base_task', choices=['Admissions', 'AdmissionsNames', 'Hiring', 'HiringNames'], default='AdmissionsNames', help='The base task to generate data for.')
parser.add_argument('--src_task', choices=['Admissions', 'AdmissionsNames', 'Hiring', 'HiringNames'], default='AdmissionsNames', help='The source task to generate data for.')
parser.add_argument('--n_samples', type=int, default=400, help='Number of samples')

parser.add_argument('--base_template_path', type=str, default=None, help='Path to the base data template')
parser.add_argument('--src_template_path', type=str, default=None, help='Path to the source data template')
parser.add_argument('--save_path', type=str, default='./', help='Path to save the data')

args = parser.parse_args()
num_samples = args.n_samples
base_task = args.base_task
src_task = args.src_task
base_template_path = args.base_template_path
src_template_path = args.src_template_path
save_path = args.save_path

os.makedirs(save_path, exist_ok=True)

if base_task == 'Admissions':
    base_settings = ADMISSIONS_SETTINGS
    base_ds_type = 'admissions_short'
elif base_task == 'AdmissionsNames':
    base_settings = ADMISSIONS_NAMES_SETTINGS
    base_ds_type = 'admissions_names'
elif base_task == 'Hiring':
    base_settings = HIRING_SETTINGS
    base_ds_type = 'hire_dec_eval'
elif base_task == 'HiringNames':
    base_settings = HIRING_NAMES_SETTINGS
    base_ds_type = 'hiring_names'

if src_task == 'Admissions':
    src_settings = ADMISSIONS_SETTINGS
    src_ds_type = 'admissions_short'
elif src_task == 'AdmissionsNames':
    src_settings = ADMISSIONS_NAMES_SETTINGS
    src_ds_type = 'admissions_names'
elif src_task == 'Hiring':
    src_settings = HIRING_SETTINGS
    src_ds_type = 'hire_dec_eval'
elif src_task == 'HiringNames':
    src_settings = HIRING_NAMES_SETTINGS
    src_ds_type = 'hiring_names'

races = base_settings['race']

base_template = open(base_template_path).read()
src_template = open(src_template_path).read()

base_profiles = [sample_one(base_settings) for _ in range(num_samples)]
src_profiles = [sample_one(src_settings) for _ in range(num_samples * len(races))]

base_profiles_controlled = []
if base_task == 'Admissions' or base_task == 'Hiring':
    for race in races:
        for profile in base_profiles:
            profile_ = profile.copy()
            profile_['race'] = race
            base_profiles_controlled.append(profile_)
elif base_task == 'AdmissionsNames' or base_task == 'HiringNames':
    names = base_settings['name']
    for race in races:
        race_names = names[race]
        for profile in base_profiles:
            profile_ = profile.copy()
            profile_['race'] = race
            profile_['name'] = random.choice(race_names)
            base_profiles_controlled.append(profile_)

assert len(base_profiles_controlled) == len(src_profiles)

base_prompts = [format_prompt(base_template, profile, dataset=base_ds_type) 
                for profile in base_profiles_controlled]
src_prompts = [format_prompt(src_template, profile, dataset=src_ds_type)
                for profile in src_profiles]

df_data = pd.DataFrame(base_profiles_controlled)
df_data['base'] = base_prompts
df_data['source'] = src_prompts
df_data['base_label'] = np.zeros(len(df_data))
df_data['src_label'] = np.zeros(len(df_data))

df_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)
