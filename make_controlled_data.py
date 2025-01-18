import argparse
import os
import random
import pandas as pd
from utils import format_prompt, ADMISSIONS_SETTINGS, ADMISSIONS_NAMES_SETTINGS, HIRING_SETTINGS, HIRING_NAMES_SETTINGS, sample_one

parser = argparse.ArgumentParser(description="Generate controlled data for debiasing.")
parser.add_argument('--task', choices=['Admissions', 'AdmissionsNames', 'Hiring', 'HiringNames'], default='AdmissionsNames', help='The task to generate data for.')
parser.add_argument('--num_samples', type=int, default=400, help='Number of samples')
parser.add_argument('--random_state', type=int, default=None, help='Random state for reproducibility')
parser.add_argument('--base_template_path', type=str, default=None, help='Path to the base data template')
parser.add_argument('--src_template_path', type=str, default=None, help='Path to the source data template')
parser.add_argument('--save_path', type=str, default='./', help='Path to save the data')

args = parser.parse_args()
num_samples = args.num_samples
random_state = args.random_state
task = args.task
base_template_path = args.base_template_path
src_template_path = args.src_template_path
save_path = args.save_path

os.makedirs(save_path, exist_ok=True)

if task == 'Admissions':
    base_settings = src_settings = ADMISSIONS_SETTINGS
    base_ds_type = src_ds_type = 'admissions_short'
elif task == 'AdmissionsNames':
    base_settings = ADMISSIONS_NAMES_SETTINGS
    src_settings = ADMISSIONS_SETTINGS
    base_ds_type = 'admissions_names'
    src_ds_type = 'admissions_short'
elif task == 'Hiring':
    base_settings = src_settings = HIRING_SETTINGS
    base_ds_type = src_ds_type = 'hire_dec_eval'
elif task == 'HiringNames':
    base_settings = HIRING_NAMES_SETTINGS
    src_settings = HIRING_SETTINGS
    base_ds_type = 'hiring_names'
    src_ds_type = 'hire_dec_eval'

races = base_settings['race']

base_template = open(base_template_path).read()
src_template = open(src_template_path).read()

base_profiles = [sample_one(base_settings) for _ in range(num_samples)]
src_profiles = [sample_one(src_settings) for _ in range(num_samples * len(races))]

base_profiles_controlled = []
if task == 'Admissions' or task == 'Hiring':
    for race in races:
        for profile in base_profiles:
            profile_ = profile.copy()
            profile_['race'] = race
            base_profiles_controlled.append(profile_)
elif task == 'AdmissionsNames' or task == 'HiringNames':
    names = base_settings['name']
    for race in races:
        race_names = names[race]
        for profile in base_profiles:
            profile_ = profile.copy()
            profile_['race'] = race
            profile_['name'] = random.choice(race_names)
            base_profiles_controlled.append(profile_)

assert len(base_profiles_controlled) == num_samples * len(races)

base_prompts = [format_prompt(base_template, profile, dataset=base_ds_type) 
                for profile in base_profiles_controlled]
src_prompts = [format_prompt(src_template, profile, dataset=src_ds_type)
                for profile in src_profiles]

df_data = pd.DataFrame(base_profiles_controlled)
df_data['base'] = base_prompts
df_data['source'] = src_prompts

df_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)
