from typing import Literal

import torch
import numpy as np

import sys
sys.path.append('../pyvene/')
from pyvene.models.basic_utils import sigmoid_boundary


HIRE_DEC_NAMES_SETTINGS = {
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'name': {
        "White": [
            "Abbey", "Abby", "Ansley", "Bailey", "Baylee", "Beth", "Caitlin", "Carley", "Carly", "Colleen",
            "Dixie", "Ginger", "Haley", "Hayley", "Heather", "Holli", "Holly", "Jane", "Jayne", "Jenna",
            "Jill", "Jodi", "Kaleigh", "Kaley", "Kari", "Katharine", "Kathleen", "Kathryn", "Kayleigh",
            "Lauri", "Laurie", "Leigh", "Lindsay", "Lori", "Luann", "Lynne", "Mandi", "Marybeth", "Mckenna",
            "Meghan", "Meredith", "Misti", "Molly", "Patti", "Sue", "Susan", "Susannah", "Susanne",
            "Suzanne", "Svetlana", "Bart", "Beau", "Braden", "Bradley", "Bret", "Brett", "Brody", "Buddy",
            "Cade", "Carson", "Cody", "Cole", "Colton", "Conner", "Connor", "Conor", "Cooper", "Dalton",
            "Dawson", "Doyle", "Dustin", "Dusty", "Gage", "Graham", "Grayson", "Gregg", "Griffin", "Hayden",
            "Heath", "Holden", "Hoyt", "Hunter", "Jack", "Jody", "Jon", "Lane", "Logan", "Parker",
            "Reed", "Reid", "Rhett", "Rocco", "Rusty", "Salvatore", "Scot", "Scott", "Stuart", "Tanner",
            "Tucker", "Wyatt"
        ],
        "Black": [
            "Amari", "Aretha", "Ashanti", "Ayana", "Ayanna", "Chiquita", "Demetria", "Eboni", "Ebony", "Essence",
            "Iesha", "Imani", "Jalisa", "Khadijah", "Kierra", "Lakeisha", "Lakesha", "Lakeshia", "Lakisha",
            "Lashanda", "Lashonda", "Latanya", "Latasha", "Latonia", "Latonya", "Latoya", "Latrice", "Nakia",
            "Precious", "Queen", "Sade", "Shalonda", "Shameka", "Shamika", "Shaneka", "Shanice", "Shanika",
            "Shaniqua", "Shante", "Sharonda", "Shawanda", "Tameka", "Tamia", "Tamika", "Tanesha", "Tanika",
            "Tawanda", "Tierra", "Tyesha", "Valencia", "Akeem", "Alphonso", "Antwan", "Cedric", "Cedrick",
            "Cornell", "Cortez", "Darius", "Darrius", "Davon", "Deandre", "Deangelo", "Demarcus", "Demario",
            "Demetrice", "Demetrius", "Deonte", "Deshawn", "Devante", "Devonte", "Donte", "Frantz", "Jabari",
            "Jalen", "Jamaal", "Jamar", "Jamel", "Jaquan", "Jarvis", "Javon", "Jaylon", "Jermaine", "Kenyatta",
            "Keon", "Lamont", "Lashawn", "Malik", "Marquis", "Marquise", "Raheem", "Rashad", "Roosevelt",
            "Shaquille", "Stephon", "Sylvester", "Tevin", "Trevon", "Tyree", "Tyrell", "Tyrone"
        ],
        "Latino": [
            "Alba", "Alejandra", "Alondra", "Amparo", "Aura", "Beatriz", "Belkis", "Blanca", "Caridad",
            "Dayana", "Dulce", "Elba", "Esmeralda", "Flor", "Graciela", "Guadalupe", "Haydee", "Iliana",
            "Ivelisse", "Ivette", "Ivonne", "Juana", "Julissa", "Lissette", "Luz", "Magaly", "Maribel",
            "Maricela", "Mariela", "Marisol", "Maritza", "Mayra", "Migdalia", "Milagros", "Mireya",
            "Mirta", "Mirtha", "Nereida", "Nidia", "Noemi", "Odalys", "Paola", "Rocio", "Viviana",
            "Xiomara", "Yadira", "Yanet", "Yesenia", "Zoila", "Zoraida", "Agustin", "Alejandro", "Alvaro",
            "Andres", "Anibal", "Arnaldo", "Camilo", "Cesar", "Diego", "Edgardo", "Eduardo", "Efrain",
            "Esteban", "Francisco", "Gerardo", "German", "Gilberto", "Gonzalo", "Guillermo", "Gustavo",
            "Hector", "Heriberto", "Hernan", "Humberto", "Jairo", "Javier", "Jesus", "Jorge", "Jose",
            "Juan", "Julio", "Lazaro", "Leonel", "Luis", "Mauricio", "Miguel", "Moises", "Norberto",
            "Octavio", "Osvaldo", "Pablo", "Pedro", "Rafael", "Ramiro", "Raul", "Reinaldo", "Rigoberto",
            "Santiago", "Santos", "Wilfredo"
        ],
        "Asian": [
            "Li Wei", "Wen Cheng", "Ming Hao", "Xiao Long", "Chao Feng", "Jie Ming", "Ping An", "Qiang Lei", "Jun Jie", "Zhi Hao",
            "Anh", "Duc", "Minh", "Tuan", "Huy", "Khanh", "Bao", "Long", "Quang", "Phuc", "Chen Wei", "Bo Tao", "Guang", "Hoang", 
            "Jisung", "Hyun", "Minjun", "Jiho", "Kyung", "Dae", "Sangwoo", "Jinwoo", "Youngho", "Yong", "Ai Mei", "Xia Lin",
            "Haruto", "Ren", "Akira", "Kaito", "Yuto", "Riku", "Hiro", "Naoki", "Shota", "Sora", "Taeyang", "Donghyun", "Lan Anh",
            "Mei Ling", "Xiao Min", "Lian Jie", "Hong Yu", "Fang Zhi", "Ying Yue", "Wei Ning", "Lan Xi", "Hui Fang", "Ming Zhu",
            "Jisoo", "Minji", "Hana", "Yuna", "Eunji", "Seojin", "Hyejin", "Soojin", "Sunhee", "Miyoung", "Haeun", "Yeji", "Mio",
            "Chi", "Linh", "Ngoc", "Phuong", "Thao", "Thanh", "Hoa", "Huong", "Trang", "Diep", "Quoc", "Dat", "Li Na", "Joon",
            "Sakura", "Yui", "Aoi", "Eri", "Mei", "Kaori", "Rina", "Yuki", "Saki", "Reina", "Mai", "Thuy", "Minseo", "Yoshi"
        ]
    },
    'experience': np.arange(0, 21, step=1),
    'degree': ['High school', 'Computer Science B.S.', 'Computer Science M.S.', 'Computer Science Ph.D.'],
    'coding': np.arange(0, 6, step=1),
}

HIRE_DEC_EVAL_SETTINGS = {
    'role' : [
        "CTO", "Physician", "Secretary", "Carpenter", "Cashier", "Receptionist",
        "Software Engineer", "Data Scientist", "Marketing Manager", "Sales Representative",
        "Teacher", "Nurse", "Accountant", "Graphic Designer", "Plumber",
        "Electrician", "Lawyer", "Architect", "Chef", "Journalist",
        "Product Manager", "Project Manager", "Financial Analyst", "Civil Engineer", 
        "Pharmacist", "Social Worker", "Web Developer", "HR Specialist", 
        "Operations Manager", "Dentist", "UX Designer", "Mechanical Engineer", 
        "Translator", "Librarian", "Pilot", "Veterinarian", 
        "Interior Designer", "IT Support Specialist", "Event Planner", "Real Estate Agent"
    ],
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'experience': np.arange(0, 21, step=1),
    'degree': ['High school', 'College', "Master's", 'Ph.D.'],
    'referrals': [0, 1, 2, 3]
}

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
            elif key == 'name':
                race = candidate['race']
                candidate['name'] = random.choice(settings['name'][race])
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


def format_prompt(template, candidate, 
                  dataset: Literal['admissions_full', 
                                 'admissions_short', 
                                 'hiring_short', 
                                 'hire_dec_eval', 
                                 'hire_dec_names'] = 'admissions_short'):
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
            num_letters = candidate['num_letters'],
            university = candidate['uni'],
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
    elif dataset == 'hire_dec_names':
        prompt = template.format(
            name = candidate['name'],
            exp = candidate['experience'],
            degree = candidate['degree'],
            coding = candidate['coding']
        )
    return prompt


"""
Given an Intervenable object with BDAS intervention, 
return its rotation matrix and boundary masks
"""
def get_bdas_params(intervenable):
    key = list(intervenable.interventions.keys())[0]
    intervention = intervenable.interventions[key][0]
    rotate_layer = intervention.rotate_layer
    Q = rotate_layer.parametrizations.weight.original
    
    intervention_boundaries = intervention.intervention_boundaries
    intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
    
    boundary_mask = sigmoid_boundary(
        intervention.intervention_population, 
        0.,
        intervention_boundaries[0] * 4096,
        intervention.temperature
    )
    
    return intervention, Q, boundary_mask


"""
A wrapper to run inference on <model> with <tokenizer>.
Accepts a list of strings and returns a list of strings.
"""
def llm_predict(model, tokenizer, device, input_batch, 
                generate=False, gen_length=None):
    input_ids = tokenizer(input_batch, 
                          return_tensors="pt", 
                          padding=True).to(device)
    input_len = input_ids['input_ids'].shape[1]

    with torch.no_grad():
        if generate:
            output_ids = model.generate(**input_ids, 
                                        max_length=input_len+gen_length)
            output_preds = tokenizer.batch_decode(output_ids[:, input_len:], 
                                                  skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=False)
        else:
            output_batch = model(**input_ids)
            output_ids = output_batch['logits'][:, -1, :].argmax(dim=-1)
            output_preds = tokenizer.batch_decode(output_ids, 
            skip_special_tokens=True)
            
    return output_preds