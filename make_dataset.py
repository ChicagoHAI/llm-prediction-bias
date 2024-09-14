import numpy as np
import random
import argparse
import os
from datasets import Dataset
import pandas as pd
from typing import Union

HIRING_SETTINGS_SHORT = {
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'experience': np.arange(0, 21, step=1),
    'degree': ['High school', 'Computer Science B.S.', 'Computer Science M.S.', 'Computer Science Ph.D.'],
    'coding': np.arange(0, 6, step=1),
}

BIOS_SETTINGS_SHORT = {
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'gpa': np.arange(1.0, 4.01, step=0.01),
    'num_ecs': np.arange(0, 9, step=1),
    'num_letters': [0, 1, 2, 3],
}

BIOS_SETTINGS = {
    'gender': ['male', 'female'],
    'race': ['white', 'black', 'latino', 'asian'],
    'income': [50, 100, 200, 400],
    'geography': ['rural America', 'urban America', 'outside the U.S.'],
    'school': ['private', 'public'],
    'gpa': [2.0, 3.0, 3.5, 3.8, 4.0],
    'sat': [1200, 1300, 1400, 1500, 1550, 1600],
    'num_ecs': [1, 2, 4, 8],
    'character_index': [1, 2, 3], # should there be a smart index?
    'letters_quality': {
        1: ['2 weak', '1 good, 1 weak'],
        2: ['1 strong, 1 weak', '2 good'],
        3: ['1 strong, 1 good', '2 strong'],
    },
    'topic': {
        1: [
            'learning to cook basic meals through following online recipes',
            'learning to knit scarves through watching YouTube tutorials',
            'learning to skateboard through practicing in the neighborhood park',
            'learning to manage personal finances through budgeting my allowance',
            'learning to take care of pets through owning a goldfish or hamster',
        ],
        2: [
            'learning to launch a startup through entrepreneurial ventures',
            'learning to navigate uncharted waters through solo sailing adventures',
            'learning to innovate in sustainable design through eco-friendly projects',
            'learning to excel in robotics through a state competition',
            'learning to analyze complex data through scientific research projects',
        ],
        3: [
            'learning to develop groundbreaking medical technology through doing summer field research',
            'learning to compose award-winning symphonies through intensive music composition programs',
            'learning to lead international humanitarian missions through nonprofit organizations',
            'learning to represent my country in the Olympic Games through exceptional athletic prowess',
            'learning to win international mathematics competitions through rigorous training and dedication',
        ]
    }
}

def is_diversity_admit(applicant):
    gpa = applicant['gpa']
    char_idx = applicant['character_index']
    race = applicant['race']
    income = applicant['income']
    geo = applicant['geography']
    
    if gpa >= 3.8 and char_idx >= 2:
        return True
    elif gpa < 3.8 and gpa >= 3.5 and char_idx >= 2 and \
    (race != 'white' or income <= 100 or geo != 'outside the U.S.'):
        return True
    elif gpa >= 3.8 and char_idx == 1 and \
    (race != 'white' or income <= 100 or geo != 'outside the U.S.'):
        return True
    elif gpa < 3.5 and gpa >= 3.0 and char_idx == 1 and \
    race == 'black':
        return True
    elif gpa < 3.0 and char_idx == 1 and \
    race == 'black' and income <= 50:
        return True
    else:
        return False

def is_diversity_admit_short(candidate):
    race = candidate['race']
    gpa = candidate['gpa']
    num_letters = candidate['num_letters']
    num_ecs = candidate['num_ecs']
    
    if race != 'White' and num_ecs >= 1:
        if num_letters >= 2 and gpa >= 3.0:
            return True
        elif num_letters == 1 and gpa >= 3.6:
            return True
        else:
            return False
    else:
        return False

"""
Sample a candidate profile
"""
def sample_one(settings, custom_stats=None):
    candidate = {}
    for key in settings:
        if custom_stats and key in custom_stats:
            candidate[key] = custom_stats[key]
        else:
            if key == 'letters_quality' or key == 'topic':
                char_index = candidate['character_index']
                candidate[key] = random.choice(settings[key][char_index])
            else:
                candidate[key] = random.choice(settings[key])
        
    if 'gender' in candidate.keys():
        if candidate['gender'] == 'male':
            candidate['pronoun'] = 'he'
            candidate['pronoun_pos'] = 'his'
        else:
            candidate['pronoun'] = 'she'
            candidate['pronoun_pos'] = 'her'
        
    # if 'num_pres' in candidate.keys():
    #     candidate['num_pres'] = random.choice(np.arange(candidate['num_ecs']))
        
    return candidate

# """
# Sample a base and source input for training DAS.
# """
# def sample_one_ctf(settings, ctf_behavior=None):
#     if ctf_behavior == None:
#         ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])
        
#     minorities = ['Black', 'Latino', 'Asian']
#     base_settings = {}
    
#     # The left variable is just {race}
#     right_true = sample_bios_short_left(settings, True)
#     right_false = sample_bios_short_left(settings, False)
    
#     # The idea is to sample base and source {race} first,
#     # then sample the same {num_letters}, {gpa}, and {num_ecs}
#     if ctf_behavior == 'f->f':
#         right_val = random.choice([True, False])
#         if right_val:
#             base_settings['num_letters'] = right_true[0]
#             base_settings['gpa'] = right_true[1]
#             base_settings['num_ecs'] = right_true[2]
            
#             base_race = src_race = 'White'
#         else:
#             base_settings['num_letters'] = right_false[0]
#             base_settings['gpa'] = right_false[1]
#             base_settings['num_ecs'] = right_false[2]
            
#             base_race = random.choice(settings['race'])
#             src_race = random.choice(settings['race'])
            
#         base_label = src_label = 'No'
#     else:
#         base_settings['num_letters'] = right_true[0]
#         base_settings['gpa'] = right_true[1]
#         base_settings['num_ecs'] = right_true[2]
        
#         if ctf_behavior == 't->t':
#             base_race = random.choice(minorities)
#             src_race = random.choice(minorities)
#             base_label = src_label = 'Yes'
#         elif ctf_behavior == 't->f':
#             base_race = random.choice(minorities)
#             src_race = 'White'
#             base_label = 'Yes'
#             src_label = 'No'
#         elif ctf_behavior == 'f->t':
#             base_race = 'White'
#             src_race = random.choice(minorities)
#             base_label = 'No'
#             src_label = 'Yes'
            
#     base_settings['race'] = base_race
#     src_settings = base_settings.copy()
#     src_settings['race'] = src_race
    
#     return base_settings, src_settings, base_label, src_label


# """
# Sample an input that leads the left variable of the short
# admissions causal model to take on <value>.
# """
# def sample_bios_short_left(settings, value: bool):
#     letters = settings['num_letters']
#     gpas = settings['gpa']
#     num_ecs = settings['num_ecs']
    
#     letter = random.choice(letters)
#     gpa = random.choice(gpas)
    
#     if value:
#         while not ((letter >= 2 and gpa >= 3.0) or \
#                     (letter == 1 and gpa >= 3.6)):
#             letter = random.choice(letters)
#             gpa = random.choice(gpas)
#         num_ec = random.choice(num_ecs[1:])
#     else:
#         while (letter >= 2 and gpa >= 3.0) or \
#                 (letter == 1 and gpa >= 3.6):
#             letter = random.choice(letters)
#             gpa = random.choice(gpas)
#         num_ec = random.choice(num_ecs)
    
#     return letter, gpa, num_ec


"""
Sample counterfactual data for aligning with (Race != White)
in the college admissions task. This is an improved version.
"""
def sample_one_ctf_admissions_race(settings, ctf_behavior=None):
    minorities = ['Black', 'Latino', 'Asian']
    
    base_profile = sample_one(settings)
    src_profile = sample_one(settings)
    
    while not is_diversity_admit_short(base_profile):
        base_profile = sample_one(settings)
        
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])
        
    if ctf_behavior == 't->t':
        src_profile['race'] = random.choice(minorities)
        base_label = src_label = 'Yes'
    elif ctf_behavior == 't->f':
        src_profile['race'] = 'White'
        base_label = 'Yes'
        src_label = 'No'
    elif ctf_behavior == 'f->t':
        base_profile['race'] = 'White'
        src_profile['race'] = random.choice(minorities)
        base_label = 'No'
        src_label = 'Yes'
    else:
        left_val = random.choice([True, False])
        if not left_val:
            # We want to make sure the left variable is false
            while is_diversity_admit_short(base_profile) or \
            base_profile['race'] == 'White':
                base_profile = sample_one(settings)
                
            base_profile['race'] = random.choice(settings['race'])
            src_profile['race'] = random.choice(settings['race'])
        else:
            base_profile['race'] = 'White'
            src_profile['race'] = 'White'        
        base_label = src_label = 'No'
        
    return base_profile, src_profile, base_label, src_label


"""
Sample a base an source input for finding an alignment
with the P := (L >= 2 AND G >= 3.0) variable, where L is
{num_letters} and G is {gpa}
"""
def sample_one_ctf_p(settings, ctf_behavior=None):
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])

    minorities = ['Black', 'Latino', 'Asian']
    good_num_ecs = settings['num_ecs'][1:] # num_ecs > 0
    base_settings = {}
    
    if ctf_behavior != 'f->f':
        base_race = random.choice(minorities)
        base_num_ecs = random.choice(good_num_ecs)
        
        base_letters = random.choice(settings['num_letters'])
        base_gpa = random.choice(settings['gpa'])
        src_letters = random.choice(settings['num_letters'])
        src_gpa = random.choice(settings['gpa'])
        
        if ctf_behavior == 't->t':
            while not (base_letters >= 2 and base_gpa >= 3.0):
                base_letters = random.choice(settings['num_letters'])
                base_gpa = random.choice(settings['gpa'])
                
            while not (src_letters >= 2 and src_gpa >= 3.0):
                src_letters = random.choice(settings['num_letters'])
                src_gpa = random.choice(settings['gpa'])
                
            base_label = src_label = 'Yes'
                
        elif ctf_behavior == 't->f':
            while not (base_letters >= 2 and base_gpa >= 3.0):
                base_letters = random.choice(settings['num_letters'])
                base_gpa = random.choice(settings['gpa'])
            
            while src_letters >= 2 and src_gpa >= 3.0:
                src_letters = random.choice(settings['num_letters'])
                src_gpa = random.choice(settings['gpa'])
                
            base_label = 'Yes'
            src_label = 'No'
                
        elif ctf_behavior == 'f->t':
            while base_letters >= 2 and base_gpa >= 3.0:
                base_letters = random.choice(settings['num_letters'])
                base_gpa = random.choice(settings['gpa'])
                
            while not (src_letters >= 2 and src_gpa >= 3.0):
                src_letters = random.choice(settings['num_letters'])
                src_gpa = random.choice(settings['gpa'])
                
            base_label = 'No'
            src_label = 'Yes'
                
    else:
        base_race = random.choice(settings['race'])
        if base_race == 'White':
            base_num_ecs = random.choice(settings['num_ecs'])
        else:
            base_num_ecs = 0
        
        base_letters = random.choice(settings['num_letters'])
        base_gpa = random.choice(settings['gpa'])
        src_letters = random.choice(settings['num_letters'])
        src_gpa = random.choice(settings['gpa'])
        
        base_label = src_label = 'No'
        
    base_settings['race'] = base_race
    base_settings['num_ecs'] = base_num_ecs
    
    src_settings = base_settings.copy()
    src_settings['num_letters'] = src_letters
    src_settings['gpa'] = src_gpa
    
    base_settings['num_letters'] = base_letters
    base_settings['gpa'] = base_gpa
    
    return base_settings, src_settings, base_label, src_label


"""
Sample a counterfactual dataset to find an alignment
with the (P = L * G) variable, where L is {num_letters}
and G is {gpa}
"""
def sample_one_ctf_L_times_G(settings, ctf_behavior=None):
    def sample_prod_letters_gpa():
        letters = random.choice(settings['num_letters'])
        gpa = random.choice(settings['gpa'])
        prod_var = letters * gpa
        return prod_var, letters, gpa
    
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])
        
    minorities = ['Black', 'Latino', 'Asian']
    good_num_ecs = settings['num_ecs'][1:] # num_ecs > 0
    base_settings = {}
    
    base_race = random.choice(minorities)
    base_num_ecs = random.choice(good_num_ecs)
    base_prod, base_letters, base_gpa = sample_prod_letters_gpa()
    src_prod, src_letters, src_gpa = sample_prod_letters_gpa()
    
    if ctf_behavior == 't->t':
        while not ((base_prod >= 6.0 and base_gpa >= 3.0) or \
                  (6 > base_prod and base_prod >= 3.6 and base_gpa >= 3.6)):
            base_prod, base_letters, base_gpa = sample_prod_letters_gpa()
        
        while not ((src_prod >= 6.0 and base_gpa >= 3.0) or \
                  (6 > src_prod and src_prod >= 3.6 and base_gpa >= 3.6)):
            src_prod, src_letters, src_gpa = sample_prod_letters_gpa()
            
        base_label = src_label = 'Yes'
        
    elif ctf_behavior == 't->f':
        while not ((base_prod >= 6.0 and base_gpa >= 3.0) or \
                  (6 > base_prod and base_prod >= 3.6 and base_gpa >= 3.6)):
            base_prod, base_letters, base_gpa = sample_prod_letters_gpa()
            
        while (src_prod >= 6.0 and base_gpa >= 3.0) or \
              (6 > src_prod and src_prod >= 3.6 and base_gpa >= 3.6):
            src_prod, src_letters, src_gpa = sample_prod_letters_gpa()
            
        base_label = 'Yes'
        src_label = 'No'
        
    elif ctf_behavior == 'f->t':
        # False because of the product variable
        while base_prod >= 3.6 or base_gpa < 3.0:
            base_prod, base_letters, base_gpa = sample_prod_letters_gpa()
            
        if base_gpa >= 3.6:
            while not (6.0 > src_prod and src_prod >= 3.6):
                src_prod, src_letters, src_gpa = sample_prod_letters_gpa()
        else:
            while src_prod <= 6.0:
                src_prod, src_letters, src_gpa = sample_prod_letters_gpa()
            
        base_label = 'No'
        src_label = 'Yes'
        
    else:
        right_var = random.choice([True, False])
        base_race = random.choice(settings['race'])
        base_num_ecs = random.choice(settings['num_ecs'])
        
        if right_var:
            base_race = random.choice(minorities)
            base_num_ecs = random.choice(good_num_ecs)
                
            while (base_prod >= 6.0 and base_gpa >= 3.0) or \
                  (6 > base_prod and base_prod >= 3.6 and base_gpa >= 3.6):
                base_prod, base_letters, base_gpa = sample_prod_letters_gpa()
            
            while (src_prod >= 6.0 and base_gpa >= 3.0) or \
                  (6 > src_prod and src_prod >= 3.6 and base_gpa >= 3.6):
                src_prod, src_letters, src_gpa = sample_prod_letters_gpa()
            
        else:
            while base_race != 'White' and base_num_ecs > 0:
                base_race = random.choice(settings['race'])
                base_num_ecs = random.choice(settings['num_ecs'])
        
        base_label = src_label = 'No'
            
    base_settings['race'] = base_race
    base_settings['num_ecs'] = base_num_ecs
    
    src_settings = base_settings.copy()
    src_settings['num_letters'] = src_letters
    src_settings['gpa'] = src_gpa
    
    base_settings['num_letters'] = base_letters
    base_settings['gpa'] = base_gpa
    
    return base_settings, src_settings, base_label, src_label


def is_diversity_hire(profile):
    higher_ed = ['Computer Science B.S.', 
                 'Computer Science M.S.', 
                 'Computer Science Ph.D.']
    higher_ed_no_ms = higher_ed[::2]
    
    def is_black_hired(degree, exp, coding):
        p = coding == 5 and ((degree in higher_ed and exp > 0) or 
                             (degree == 'High school' and exp in range(2, 16)))
        q = coding == 4 and ((degree in higher_ed_no_ms and exp > 0) or 
                             (degree == 'Computer Science M.S.' and exp in range(3, 9)))
        return p or q

    def is_latino_hired(degree, exp, coding):
        p = coding == 5 and exp > 0
        q = coding == 4 and degree in higher_ed and exp > 0
        return p or q

    def is_asian_hired(degree, exp, coding):
        p = coding == 5 and degree in higher_ed and exp > 0
        q = coding == 4 and ((degree in higher_ed_no_ms and exp > 0) or 
                             (degree == 'Computer Science M.S.' and exp in range(2, 16)))
        return p or q
    
    race = profile['race']
    degree = profile['degree']
    exp = profile['experience']
    coding = profile['coding']

    if race == 'Black':
        return is_black_hired(degree, exp, coding)
    elif race == 'Latino':
        return is_latino_hired(degree, exp, coding)
    elif race == 'Asian':
        return is_asian_hired(degree, exp, coding)
    else:
        return False


def is_diversity_hire_simple(profile):
    race = profile['race']
    degree = profile['degree']
    exp = profile['experience']
    coding = profile['coding']
    
    if race != 'White':
        if exp > 0:
            if coding == 5:
                return True
            elif coding == 4 and degree != 'High school':
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def sample_one_ctf_hiring_race(settings, ctf_behavior=None):
    minorities = ['Black', 'Latino', 'Asian']
    
    base_profile = sample_one(settings)
    src_profile = sample_one(settings)
    
    while not is_diversity_hire(base_profile):
        base_profile = sample_one(settings)
    
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 'f->f'])
        
    if ctf_behavior == 't->t':
        src_profile['race'] = random.choice(minorities)
        base_label = src_label = 'Yes'
    elif ctf_behavior == 't->f':
        src_profile['race'] = 'White'
        base_label = 'Yes'
        src_label = 'No'
    elif ctf_behavior == 'f->t':
        base_profile['race'] = 'White'
        src_profile['race'] = random.choice(minorities)
        base_label = 'No'
        src_label = 'Yes'
    else:
        left_val = random.choice([True, False])
        if not left_val:
            while is_diversity_hire(base_profile):
                base_profile = sample_one(settings)
            while is_diversity_hire(src_profile):
                src_profile = sample_one(settings)
        else:
            base_profile['race'] = 'White'
            src_profile['race'] = 'White'        
        base_label = src_label = 'No'
        
    return base_profile, src_profile, base_label, src_label


def sample_one_ctf_hiring_race_simple(settings, ctf_behavior=None):
    minorities = ['Black', 'Latino', 'Asian']
    
    base_profile = sample_one(settings)
    src_profile = sample_one(settings)
    
    while not is_diversity_hire_simple(base_profile):
        base_profile = sample_one(settings)
        
    if ctf_behavior == None:
        ctf_behavior = random.choice(['t->t', 't->f', 'f->t', 't->t'])
        
    if ctf_behavior == 't->t':
        src_profile['race'] = random.choice(minorities)
        base_label = src_label = 'Yes'
    elif ctf_behavior == 't->f':
        src_profile['race'] = 'White'
        base_label = 'Yes'
        src_label = 'No'
    elif ctf_behavior == 'f->t':
        base_profile['race'] = 'White'
        src_profile['race'] = random.choice(minorities)
        base_label = 'No'
        src_label = 'Yes'
    else:
        left_val = random.choice([True, False])
        if not left_val:
            # We want to make sure the left variable is false
            while is_diversity_hire(base_profile) or \
            base_profile['race'] == 'White':
                base_profile = sample_one(settings)
                
            base_profile['race'] = random.choice(settings['race'])
            src_profile['race'] = random.choice(settings['race'])
        else:
            base_profile['race'] = 'White'
            src_profile['race'] = 'White'        
        base_label = src_label = 'No'
        
    return base_profile, src_profile, base_label, src_label


def format_prompt(template, candidate, 
                  dataset: Union['admissions_full', 
                                 'admissions_short', 
                                 'hiring_short', 
                                 'hire_dec_eval'] = 'admissions_short'):
    if dataset == 'admissions_full':
        prompt = template.format(
            pronoun_pos = candidate['pronoun_pos'],
            pronoun = candidate['pronoun'],
            gender = candidate['gender'],
            race = candidate['race'],
            income = candidate['income'],
            geography = candidate['geography'],
            school = candidate['school'],
            gpa = candidate['gpa'],
            sat = candidate['sat'],
            num_ecs = candidate['num_ecs'],
            # num_pres = candidate['num_pres'],
            letters_quality = candidate['letters_quality'],
            topic = candidate['topic']
        )
    elif dataset == 'admissions_short':
        prompt = template.format(
            race = candidate['race'],
            gpa = candidate['gpa'],
            num_ecs = candidate['num_ecs'],
            num_letters = candidate['num_letters']
        )
    elif dataset == 'hiring_short':
        prompt = template.format(
            race = candidate['race'],
            exp = candidate['experience'],
            degree = candidate['degree'],
            coding = candidate['coding']
        )
    elif dataset == 'hire_dec_eval':
        prompt = template.format(
            role = candidate['role'],
            race = candidate['race'],
            exp = candidate['experience'],
            degree = candidate['degree'],
            referrals = candidate['referrals']
        )
    return prompt

"""
Tokenize the label. The returned token index is somewhat
arbitrary because it depends on the tokenizer. In this case
it is the LLaMA tokenizer.
"""
def format_label(label_eng):
    if label_eng == 'Yes':
        return 8241
    else:
        return 3782


dataset_type_dict = {
    "admissions_race": BIOS_SETTINGS_SHORT,
    "admissions_p-var": BIOS_SETTINGS_SHORT,
    "admissions_prod-var": BIOS_SETTINGS_SHORT,
    "hiring_race": HIRING_SETTINGS_SHORT,
}

dataset_format_dict = {
    "admissions_race": "admissions_short",
    "admissions_p-var": "admissions_short",
    "admissions_prod-var": "admissions_short",
    "hiring_race": "hiring_short",
}

ctf_function_dict = {
    "admissions_race": sample_one_ctf_admissions_race,
    "admissions_p-var": sample_one_ctf_p,
    "admissions_prod-var": sample_one_ctf_L_times_G,
    "hiring_race": sample_one_ctf_hiring_race_simple,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_size", help="Specify the size of the dataset.",
    default=1000, type=int)
    parser.add_argument("--base_template_path", 
    help="""Path to the prompt template for the base.""")
    parser.add_argument("--source_template_path", 
    help="""Path to the prompt template for the base.""")
    parser.add_argument("--base_dataset_type", 
    help="""Specify which causal variable you are aligning with. Options:
    - admissions_race
    - admissions_p-var
    - admissions_prod-var
    - hiring_race"""
    )
    parser.add_argument("--source_dataset_type", help="Same as --base_dataset_type.")
    parser.add_argument("--save_path", help="""Path to save the resulting dataset. 
                                            Should end in a directory.""")

    args = parser.parse_args()

    base_ds_type = args.base_dataset_type
    src_ds_type = args.source_dataset_type

    ds_size = args.dataset_size
    save_path = args.save_path

    # Making a dataset
    template_base = open(args.base_template_path).read()
    template_src = open(args.source_template_path).read()

    SETTINGS_BASE = dataset_type_dict[base_ds_type]
    SETTINGS_SRC = dataset_type_dict[src_ds_type]
    sample_one_ctf_task_base = ctf_function_dict[base_ds_type]

    ctf_task_base_examples = [sample_one_ctf_task_base(SETTINGS_BASE) for _ in range(ds_size)]
    src_task_examples = [sample_one(SETTINGS_SRC) for _ in range(ds_size)]

    # Only cross-task race variable is supported for now
    if base_ds_type != src_ds_type and \
    base_ds_type in ["admissions_race", "hiring_race"] and \
    src_ds_type in ["admissions_race", "hiring_race"]:

        for i in range(ds_size):
            base_task_src_race = ctf_task_base_examples[i][1]['race']
            src_task_examples[i]['race'] = base_task_src_race
            
            new_ctf_example = (ctf_task_base_examples[i][0], 
                            src_task_examples[i], 
                            ctf_task_base_examples[i][2], 
                            ctf_task_base_examples[i][3])
            
            ctf_task_base_examples[i] = new_ctf_example

    dataset_dict = {
        'base': [format_prompt(template_base, ex[0], dataset=dataset_format_dict[base_ds_type]) 
        for ex in ctf_task_base_examples],
        'source': [format_prompt(template_src, ex[1], dataset=dataset_format_dict[src_ds_type]) 
        for ex in ctf_task_base_examples],
        'base_label': [format_label(ex[2]) for ex in ctf_task_base_examples],
        'src_label': [format_label(ex[3]) for ex in ctf_task_base_examples]
    }

    dataset_all = Dataset.from_dict(dataset_dict)
    ds_train_test = dataset_all.train_test_split(test_size=0.2)
    ds_test = ds_train_test['test']
    ds_train_dev = ds_train_test['train'].train_test_split(test_size=0.2)
    ds_train = ds_train_dev['train']
    ds_dev = ds_train_dev['test']

    df_train = pd.DataFrame(ds_train)
    df_dev = pd.DataFrame(ds_dev)
    df_test = pd.DataFrame(ds_test)

    os.makedirs(save_path, exist_ok=True)
    df_train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    df_dev.to_csv(os.path.join(save_path, "dev.csv"), index=False)
    df_test.to_csv(os.path.join(save_path, "test.csv"), index=False)