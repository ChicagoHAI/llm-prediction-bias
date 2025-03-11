from typing import Literal
import random
import os

import torch
import numpy as np

import sys
sys.path.append('../pyvene/')
from pyvene.models.basic_utils import sigmoid_boundary
import pyvene as pv

NAMES = {
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
}

NAMES_GENDER = {
    "White": {
        "female": [
            "Abbey", "Abby", "Ansley", "Bailey", "Baylee", "Beth", "Caitlin", "Carley", "Carly", "Colleen",
            "Dixie", "Ginger", "Haley", "Hayley", "Heather", "Holli", "Holly", "Jane", "Jayne", "Jenna",
            "Jill", "Jodi", "Kaleigh", "Kaley", "Kari", "Katharine", "Kathleen", "Kathryn", "Kayleigh",
            "Lauri", "Laurie", "Leigh", "Lindsay", "Lori", "Luann", "Lynne", "Mandi", "Marybeth", "Mckenna",
            "Meghan", "Meredith", "Misti", "Molly", "Patti", "Sue", "Susan", "Susannah", "Susanne",
            "Suzanne", "Svetlana"
        ],
        "male": [
            "Bart", "Beau", "Braden", "Bradley", "Bret", "Brett", "Brody", "Buddy",
            "Cade", "Carson", "Cody", "Cole", "Colton", "Conner", "Connor", "Conor", "Cooper", "Dalton",
            "Dawson", "Doyle", "Dustin", "Dusty", "Gage", "Graham", "Grayson", "Gregg", "Griffin", "Hayden",
            "Heath", "Holden", "Hoyt", "Hunter", "Jack", "Jody", "Jon", "Lane", "Logan", "Parker",
            "Reed", "Reid", "Rhett", "Rocco", "Rusty", "Salvatore", "Scot", "Scott", "Stuart", "Tanner",
            "Tucker", "Wyatt"
        ]
    },
    "Black": {
        "female": [
            "Amari", "Aretha", "Ashanti", "Ayana", "Ayanna", "Chiquita", "Demetria", "Eboni", "Ebony", "Essence",
            "Iesha", "Imani", "Jalisa", "Khadijah", "Kierra", "Lakeisha", "Lakesha", "Lakeshia", "Lakisha",
            "Lashanda", "Lashonda", "Latanya", "Latasha", "Latonia", "Latonya", "Latoya", "Latrice", "Nakia",
            "Precious", "Queen", "Sade", "Shalonda", "Shameka", "Shamika", "Shaneka", "Shanice", "Shanika",
            "Shaniqua", "Shante", "Sharonda", "Shawanda", "Tameka", "Tamia", "Tamika", "Tanesha", "Tanika",
            "Tawanda", "Tierra", "Tyesha", "Valencia"
        ],
        "male": [
            "Akeem", "Alphonso", "Antwan", "Cedric", "Cedrick",
            "Cornell", "Cortez", "Darius", "Darrius", "Davon", "Deandre", "Deangelo", "Demarcus", "Demario",
            "Demetrice", "Demetrius", "Deonte", "Deshawn", "Devante", "Devonte", "Donte", "Frantz", "Jabari",
            "Jalen", "Jamaal", "Jamar", "Jamel", "Jaquan", "Jarvis", "Javon", "Jaylon", "Jermaine", "Kenyatta",
            "Keon", "Lamont", "Lashawn", "Malik", "Marquis", "Marquise", "Raheem", "Rashad", "Roosevelt",
            "Shaquille", "Stephon", "Sylvester", "Tevin", "Trevon", "Tyree", "Tyrell", "Tyrone"
        ]
    },
    "Latino": {
        "female": [
            "Alba", "Alejandra", "Alondra", "Amparo", "Aura", "Beatriz", "Belkis", "Blanca", "Caridad",
            "Dayana", "Dulce", "Elba", "Esmeralda", "Flor", "Graciela", "Guadalupe", "Haydee", "Iliana",
            "Ivelisse", "Ivette", "Ivonne", "Juana", "Julissa", "Lissette", "Luz", "Magaly", "Maribel",
            "Maricela", "Mariela", "Marisol", "Maritza", "Mayra", "Migdalia", "Milagros", "Mireya",
            "Mirta", "Mirtha", "Nereida", "Nidia", "Noemi", "Odalys", "Paola", "Rocio", "Viviana",
            "Xiomara", "Yadira", "Yanet", "Yesenia", "Zoila", "Zoraida"
        ],
        "male": [
            "Agustin", "Alejandro", "Alvaro",
            "Andres", "Anibal", "Arnaldo", "Camilo", "Cesar", "Diego", "Edgardo", "Eduardo", "Efrain",
            "Esteban", "Francisco", "Gerardo", "German", "Gilberto", "Gonzalo", "Guillermo", "Gustavo",
            "Hector", "Heriberto", "Hernan", "Humberto", "Jairo", "Javier", "Jesus", "Jorge", "Jose",
            "Juan", "Julio", "Lazaro", "Leonel", "Luis", "Mauricio", "Miguel", "Moises", "Norberto",
            "Octavio", "Osvaldo", "Pablo", "Pedro", "Rafael", "Ramiro", "Raul", "Reinaldo", "Rigoberto",
            "Santiago", "Santos", "Wilfredo"
        ]
    },
    "Asian": {
        "female": [
            "Ai Mei", "Xia Lin", "Lan Anh", "Mei Ling", "Xiao Min", "Lian Jie", "Hong Yu", "Fang Zhi", "Ying Yue",
            "Wei Ning", "Lan Xi", "Hui Fang", "Ming Zhu", "Jisoo", "Minji", "Hana", "Yuna", "Eunji", "Seojin",
            "Hyejin", "Soojin", "Sunhee", "Miyoung", "Haeun", "Yeji", "Mio", "Chi", "Linh", "Ngoc", "Phuong",
            "Thao", "Thanh", "Hoa", "Huong", "Trang", "Diep", "Li Na", "Sakura", "Yui", "Aoi", "Eri", "Mei",
            "Kaori", "Rina", "Yuki", "Saki", "Reina", "Mai", "Thuy", "Minseo"
        ],
        "male": [
            "Li Wei", "Wen Cheng", "Ming Hao", "Xiao Long", "Chao Feng", "Jie Ming", "Ping An", "Qiang Lei",
            "Jun Jie", "Zhi Hao", "Anh", "Duc", "Minh", "Tuan", "Huy", "Khanh", "Bao", "Long", "Quang", "Phuc",
            "Chen Wei", "Bo Tao", "Guang", "Hoang", "Jisung", "Hyun", "Minjun", "Jiho", "Kyung", "Dae",
            "Sangwoo", "Jinwoo", "Youngho", "Yong", "Haruto", "Ren", "Akira", "Kaito", "Yuto", "Riku", "Hiro",
            "Naoki", "Shota", "Sora", "Taeyang", "Donghyun", "Quoc", "Dat", "Joon", "Yoshi"
        ]
    }
}

NAMES_CAP = {
    "White": [
        "ABBEY", "ABBY", "ANSLEY", "BAILEY", "BAYLEE", "BETH", "CAITLIN", "CARLEY", "CARLY", "COLLEEN",
        "DIXIE", "GINGER", "HALEY", "HAYLEY", "HEATHER", "HOLLI", "HOLLY", "JANE", "JAYNE", "JENNA",
        "JILL", "JODI", "KALEIGH", "KALEY", "KARI", "KATHARINE", "KATHLEEN", "KATHRYN", "KAYLEIGH",
        "LAURI", "LAURIE", "LEIGH", "LINDSAY", "LORI", "LUANN", "LYNNE", "MANDI", "MARYBETH", "MCKENNA",
        "MEGHAN", "MEREDITH", "MISTI", "MOLLY", "PATTI", "SUE", "SUSAN", "SUSANNAH", "SUSANNE",
        "SUZANNE", "SVETLANA", "BART", "BEAU", "BRADEN", "BRADLEY", "BRET", "BRETT", "BRODY", "BUDDY",
        "CADE", "CARSON", "CODY", "COLE", "COLTON", "CONNER", "CONNOR", "CONOR", "COOPER", "DALTON",
        "DAWSON", "DOYLE", "DUSTIN", "DUSTY", "GAGE", "GRAHAM", "GRAYSON", "GREGG", "GRIFFIN", "HAYDEN",
        "HEATH", "HOLDEN", "HOYT", "HUNTER", "JACK", "JODY", "JON", "LANE", "LOGAN", "PARKER",
        "REED", "REID", "RHETT", "ROCCO", "RUSTY", "SALVATORE", "SCOT", "SCOTT", "STUART", "TANNER",
        "TUCKER", "WYATT"
    ],
    "Black": [
        "AMARI", "ARETHA", "ASHANTI", "AYANA", "AYANNA", "CHIQUITA", "DEMETRIA", "EBONI", "EBONY", "ESSENCE",
        "IESHA", "IMANI", "JALISA", "KHADIJAH", "KIERRA", "LAKEISHA", "LAKESHA", "LAKESHIA", "LAKISHA",
        "LASHANDA", "LASHONDA", "LATANYA", "LATASHA", "LATONIA", "LATONYA", "LATOYA", "LATRICE", "NAKIA",
        "PRECIOUS", "QUEEN", "SADE", "SHALONDA", "SHAMEKA", "SHAMIKA", "SHANEKA", "SHANICE", "SHANIKA",
        "SHANIQUA", "SHANTE", "SHARONDA", "SHAWANDA", "TAMEKA", "TAMIA", "TAMIKA", "TANESHA", "TANIKA",
        "TAWANDA", "TIERRA", "TYESHA", "VALENCIA", "AKEEM", "ALPHONSO", "ANTWAN", "CEDRIC", "CEDRICK",
        "CORNELL", "CORTEZ", "DARIUS", "DARRIUS", "DAVON", "DEANDRE", "DEANGELO", "DEMARCUS", "DEMARIO",
        "DEMETRICE", "DEMETRIUS", "DEONTE", "DESHAWN", "DEVANTE", "DEVONTE", "DONTE", "FRANTZ", "JABARI",
        "JALEN", "JAMAAL", "JAMAR", "JAMEL", "JAQUAN", "JARVIS", "JAVON", "JAYLON", "JERMAINE", "KENYATTA",
        "KEON", "LAMONT", "LASHAWN", "MALIK", "MARQUIS", "MARQUISE", "RAHEEM", "RASHAD", "ROOSEVELT",
        "SHAQUILLE", "STEPHON", "SYLVESTER", "TEVIN", "TREVON", "TYREE", "TYRELL", "TYRONE"
    ],
    "Latino": [
        "ALBA", "ALEJANDRA", "ALONDRA", "AMPARO", "AURA", "BEATRIZ", "BELKIS", "BLANCA", "CARIDAD",
        "DAYANA", "DULCE", "ELBA", "ESMERALDA", "FLOR", "GRACIELA", "GUADALUPE", "HAYDEE", "ILIANA",
        "IVELISSE", "IVETTE", "IVONNE", "JUANA", "JULISSA", "LISSETTE", "LUZ", "MAGALY", "MARIBEL",
        "MARICELA", "MARIELA", "MARISOL", "MARITZA", "MAYRA", "MIGDALIA", "MILAGROS", "MIREYA",
        "MIRTA", "MIRTHA", "NEREIDA", "NIDIA", "NOEMI", "ODALYS", "PAOLA", "ROCIO", "VIVIANA",
        "XIOMARA", "YADIRA", "YANET", "YESENIA", "ZOILA", "ZORAIDA", "AGUSTIN", "ALEJANDRO", "ALVARO",
        "ANDRES", "ANIBAL", "ARNALDO", "CAMILO", "CESAR", "DIEGO", "EDGARDO", "EDUARDO", "EFRAIN",
        "ESTEBAN", "FRANCISCO", "GERARDO", "GERMAN", "GILBERTO", "GONZALO", "GUILLERMO", "GUSTAVO",
        "HECTOR", "HERIBERTO", "HERNAN", "HUMBERTO", "JAIRO", "JAVIER", "JESUS", "JORGE", "JOSE",
        "JUAN", "JULIO", "LAZARO", "LEONEL", "LUIS", "MAURICIO", "MIGUEL", "MOISES", "NORBERTO",
        "OCTAVIO", "OSVALDO", "PABLO", "PEDRO", "RAFAEL", "RAMIRO", "RAUL", "REINALDO", "RIGOBERTO",
        "SANTIAGO", "SANTOS", "WILFREDO"
    ],
    "Asian": [
        "LI WEI", "WEN CHENG", "MING HAO", "XIAO LONG", "CHAO FENG", "JIE MING", "PING AN", "QIANG LEI", "JUN JIE", "ZHI HAO",
        "ANH", "DUC", "MINH", "TUAN", "HUY", "KHANH", "BAO", "LONG", "QUANG", "PHUC", "CHEN WEI", "BO TAO", "GUANG", "HOANG", 
        "JISUNG", "HYUN", "MINJUN", "JIHO", "KYUNG", "DAE", "SANGWOO", "JINWOO", "YOUNGHO", "YONG", "AI MEI", "XIA LIN",
        "HARUTO", "REN", "AKIRA", "KAITO", "YUTO", "RIKU", "HIRO", "NAOKI", "SHOTA", "SORA", "TAEYANG", "DONGHYUN", "LAN ANH",
        "MEI LING", "XIAO MIN", "LIAN JIE", "HONG YU", "FANG ZHI", "YING YUE", "WEI NING", "LAN XI", "HUI FANG", "MING ZHU",
        "JISOO", "MINJI", "HANA", "YUNA", "EUNJI", "SEOJIN", "HYEJIN", "SOOJIN", "SUNHEE", "MIYOUNG", "HAEUN", "YEJI", "MIO",
        "CHI", "LINH", "NGOC", "PHUONG", "THAO", "THANH", "HOA", "HUONG", "TRANG", "DIEP", "QUOC", "DAT", "LI NA", "JOON",
        "SAKURA", "YUI", "AOI", "ERI", "MEI", "KAORI", "RINA", "YUKI", "SAKI", "REINA", "MAI", "THUY", "MINSEO", "YOSHI"
    ]
}

NAMES_LASTNAMES = {
    "White": [
        "Abbey Smith", "Abby Johnson", "Ansley Williams", "Bailey Brown", "Baylee Jones", "Beth Miller", "Caitlin Davis", "Carley Wilson", "Carly Anderson", "Colleen Taylor",
        "Dixie Smith", "Ginger Johnson", "Haley Williams", "Hayley Brown", "Heather Jones", "Holli Miller", "Holly Davis", "Jane Wilson", "Jayne Anderson", "Jenna Taylor",
        "Jill Smith", "Jodi Johnson", "Kaleigh Williams", "Kaley Brown", "Kari Jones", "Katharine Miller", "Kathleen Davis", "Kathryn Wilson", "Kayleigh Anderson", "Lauri Taylor",
        "Laurie Smith", "Leigh Johnson", "Lindsay Williams", "Lori Brown", "Luann Jones", "Lynne Miller", "Mandi Davis", "Marybeth Wilson", "Mckenna Anderson", "Meghan Taylor",
        "Meredith Smith", "Misti Johnson", "Molly Williams", "Patti Brown", "Sue Jones", "Susan Miller", "Susannah Davis", "Susanne Wilson", "Suzanne Anderson", "Svetlana Taylor",
        "Bart Smith", "Beau Johnson", "Braden Williams", "Bradley Brown", "Bret Jones", "Brett Miller", "Brody Davis", "Buddy Wilson", "Cade Anderson", "Carson Taylor",
        "Cody Smith", "Cole Johnson", "Colton Williams", "Conner Brown", "Connor Jones", "Conor Miller", "Cooper Davis", "Dalton Wilson", "Dawson Anderson", "Doyle Taylor",
        "Dustin Smith", "Dusty Johnson", "Gage Williams", "Graham Brown", "Grayson Jones", "Gregg Miller", "Griffin Davis", "Hayden Wilson", "Heath Anderson", "Holden Taylor",
        "Hoyt Smith", "Hunter Johnson", "Jack Williams", "Jody Brown", "Jon Jones", "Lane Miller", "Logan Davis", "Parker Wilson", "Reed Anderson", "Reid Taylor",
        "Rhett Smith", "Rocco Johnson", "Rusty Williams", "Salvatore Brown", "Scot Jones", "Scott Miller", "Stuart Davis", "Tanner Wilson", "Tucker Anderson", "Wyatt Taylor"
    ],
    "Black": [
        "Amari Robinson", "Aretha Washington", "Ashanti Jefferson", "Ayana Jackson", "Ayanna Carter", "Chiquita Harris", "Demetria Lewis", "Eboni Walker", "Ebony Thomas", "Essence Scott",
        "Iesha Moore", "Imani Robinson", "Jalisa Wright", "Khadijah Green", "Kierra Allen", "Lakeisha Young", "Lakesha King", "Lakeshia Hill", "Lakisha Adams", "Lashanda Baker",
        "Lashonda Nelson", "Latanya Carter", "Latasha Mitchell", "Latonia Perez", "Latonya Roberts", "Latoya Turner", "Latrice Parker", "Nakia Collins", "Precious Stewart", "Queen Jenkins",
        "Sade Russell", "Shalonda Barnes", "Shameka Murphy", "Shamika Howard", "Shaneka Long", "Shanice Bryant", "Shanika Griffin", "Shaniqua Brooks", "Shante Reed", "Sharonda Cox",
        "Shawanda Ward", "Tameka Richardson", "Tamia Butler", "Tamika Simmons", "Tanesha Foster", "Tanika Peterson", "Tawanda Graham", "Tierra Jenkins", "Tyesha Russell", "Valencia Hayes",
        "Akeem Bell", "Alphonso Coleman", "Antwan Butler", "Cedric Jenkins", "Cedrick Brooks", "Cornell Foster", "Cortez Washington", "Darius Price", "Darrius Bryant", "Davon Long",
        "Deandre Morgan", "Deangelo Russell", "Demarcus Perry", "Demario Jenkins", "Demetrice Hayes", "Demetrius Butler", "Deonte Powell", "Deshawn Simmons", "Devante Brooks", "Devonte Washington",
        "Donte Richardson", "Frantz Carter", "Jabari Reed", "Jalen Turner", "Jamaal Scott", "Jamar Parker", "Jamel Powell", "Jaquan Davis", "Jarvis Cooper", "Javon Hughes",
        "Jaylon Price", "Jermaine Jenkins", "Kenyatta Bell", "Keon Brooks", "Lamont Foster", "Lashawn Hayes", "Malik Powell", "Marquis Jenkins", "Marquise Carter", "Raheem Robinson",
        "Rashad Walker", "Roosevelt Davis", "Shaquille Robinson", "Stephon Bell", "Sylvester Jenkins", "Tevin Brooks", "Trevon Foster", "Tyree Hayes", "Tyrell Powell", "Tyrone Jenkins"
    ],
    "Latino": [
        "Alba Ramirez", "Alejandra Martinez", "Alondra Lopez", "Amparo Gonzalez", "Aura Hernandez", "Beatriz Sanchez", "Belkis Torres", "Blanca Castillo", "Caridad Ruiz", "Dayana Vega",
        "Dulce Rios", "Elba Pacheco", "Esmeralda Soto", "Flor Navarro", "Graciela Mendez", "Guadalupe Alvarez", "Haydee Chavez", "Iliana Delgado", "Ivelisse Morales", "Ivette Santiago",
        "Ivonne Fuentes", "Juana Ortiz", "Julissa Espinoza", "Lissette Peña", "Luz Solis", "Magaly Nunez", "Maribel Velez", "Maricela Herrera", "Mariela Cabrera", "Marisol Montoya",
        "Maritza Serrano", "Mayra Camacho", "Migdalia Cisneros", "Milagros Valdez", "Mireya Trujillo", "Mirta Bautista", "Mirtha Ochoa", "Nereida Carrillo", "Nidia Trejo", "Noemi Rosales",
        "Odalys Luna", "Paola Ibarra", "Rocio Cordova", "Viviana Pizarro", "Xiomara Arce", "Yadira Salinas", "Yanet Moya", "Yesenia Duran", "Zoila Calderon", "Zoraida Meza",
        "Agustin Rivera", "Alejandro Cruz", "Alvaro Mejia", "Andres Vargas", "Anibal Padilla", "Arnaldo Duarte", "Camilo Rangel", "Cesar Escobar", "Diego Figueroa", "Edgardo Palacios",
        "Eduardo Soto", "Efrain Rojas", "Esteban Peña", "Francisco Delgado", "Gerardo Ponce", "German Valdez", "Gilberto Fuentes", "Gonzalo Herrera", "Guillermo Cruz", "Gustavo Morales",
        "Hector Ramirez", "Heriberto Castro", "Hernan Rios", "Humberto Soto", "Jairo Mendez", "Javier Delgado", "Jesus Pacheco", "Jorge Morales", "Jose Herrera", "Juan Vargas",
        "Julio Castillo", "Lazaro Rios", "Leonel Aguilar", "Luis Navarro", "Mauricio Silva", "Miguel Cabrera", "Moises Rojas", "Norberto Solis", "Octavio Ponce", "Osvaldo Rangel",
        "Pablo Ortiz", "Pedro Reyes", "Rafael Fuentes", "Ramiro Velasquez", "Raul Andrade", "Reinaldo Rios", "Rigoberto Santana", "Santiago Carrillo", "Santos Morales", "Wilfredo Luna"
    ],
    "Asian": [
        "Li Wei Zhang", "Wen Cheng Liu", "Ming Hao Wong", "Xiao Long Huang", "Chao Feng Tang", "Jie Ming Lin", "Ping An Xu", "Qiang Lei Yang", "Jun Jie Zhao", "Zhi Hao Ma",
        "Anh Nguyen", "Duc Tran", "Minh Le", "Tuan Pham", "Huy Vo", "Khanh Bui", "Bao Ho", "Long Dang", "Quang Thai", "Phuc Lam", "Chen Wei Guo", "Bo Tao Zeng", "Guang Han", "Hoang Vu",
        "Jisung Kim", "Hyun Park", "Minjun Choi", "Jiho Lee", "Kyung Seo", "Dae Shin", "Sangwoo Kwon", "Jinwoo Kang", "Youngho Jeong", "Yong Son", "Ai Mei Fang", "Xia Lin Deng",
        "Haruto Sato", "Ren Ishikawa", "Akira Takahashi", "Kaito Yamamoto", "Yuto Nakamura", "Riku Sato", "Hiro Tanaka", "Naoki Suzuki", "Shota Kobayashi", "Sora Fujimoto", "Taeyang Ito", "Donghyun Kim", "Lan Anh Le",
        "Mei Ling Chen", "Xiao Min Wang", "Lian Jie Zhao", "Hong Yu Gu", "Fang Zhi Lin", "Ying Yue Chen", "Wei Ning Liu", "Lan Xi Zhang", "Hui Fang Wu", "Ming Zhu Chen",
        "Jisoo Park", "Minji Kim", "Hana Lee", "Yuna Kim", "Eunji Choi", "Seojin Lee", "Hyejin Park", "Soojin Kim", "Sunhee Lee", "Miyoung Kim", "Haeun Park", "Yeji Choi", "Mio Lee",
        "Chi Nguyen", "Linh Tran", "Ngoc Pham", "Phuong Le", "Thao Nguyen", "Thanh Pham", "Hoa Nguyen", "Huong Tran", "Trang Le", "Diep Nguyen", "Quoc Tran", "Dat Le", "Li Na Chen", "Joon Park",
        "Sakura Sato", "Yui Tanaka", "Aoi Yamamoto", "Eri Suzuki", "Mei Takahashi", "Kaori Nakamura", "Rina Sato", "Yuki Tanaka", "Saki Yamamoto", "Reina Suzuki", "Mai Nakamura", "Thuy Takahashi", "Minseo Kim", "Yoshi Sato"
    ]
}



HIRE_DEC_NAMES_SETTINGS = {
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'name': NAMES,
    'experience': np.arange(0, 21, step=1),
    'degree': ['High school', 'Computer Science B.S.', 'Computer Science M.S.', 'Computer Science Ph.D.'],
    'coding': np.arange(0, 6, step=1),
}

HIRING_SETTINGS = {
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

HIRING_NAMES_SETTINGS = {
    'race': ['White', 'Black', 'Latino', 'Asian'],
    'name': NAMES,
    # 'name': NAMES_CAP,
    # 'name': NAMES_LASTNAMES,
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

ADMISSIONS_NAMES_SETTINGS = {
    'uni': ["Harvard University", "Stanford University", "University of Chicago", "University of Alabama", "University of California, Berkeley", "University of Michigan", "University of Southern California", "Northwestern University", "University of Texas at Austin", "University of North Carolina at Chapel Hill", "Florida State University", "University of Miami", "University of Minnesota", "Howard University", "University of Wisconsin-Madison", "University of Maryland, College Park", "University of Arizona", "University of Pittsburgh", "University of Iowa", "University of Notre Dame"],
    'race': ['White', 'Black', 'Latino', 'Asian'],
    # 'gender': ['male', 'female'],
    'name': NAMES,
    # 'name': NAMES_CAP,
    # 'name': NAMES_LASTNAMES,
    # 'name': NAMES_GENDER,
    'gpa': np.arange(1.0, 4.1, step=0.1),
    'num_ecs': np.arange(0, 9, step=1),
    'num_letters': [0, 1, 2, 3],
}

ADMISSIONS_SETTINGS = {
    'uni': ["Harvard University", "Stanford University", "University of Chicago", "University of Alabama", "University of California, Berkeley", "University of Michigan", "University of Southern California", "Northwestern University", "University of Texas at Austin", "University of North Carolina at Chapel Hill", "Florida State University", "University of Miami", "University of Minnesota", "Howard University", "University of Wisconsin-Madison", "University of Maryland, College Park", "University of Arizona", "University of Pittsburgh", "University of Iowa", "University of Notre Dame"],
    'race': ['White', 'Black', 'Latino', 'Asian'],
    # 'gpa': np.arange(1.0, 4.01, step=0.01),
    'gpa': np.arange(1.0, 4.1, step=0.1),
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
# def sample_one(settings, custom_stats=None):
#     candidate = {}
#     for key in settings:
#         if custom_stats and key in custom_stats:
#             candidate[key] = custom_stats[key]
#         else:
#             if key == 'letters_quality' or key == 'topic':
#                 char_index = candidate['character_index']
#                 candidate[key] = random.choice(settings[key][char_index])
#             elif key == 'name':
#                 # race = candidate['race']
#                 # gender = candidate['gender']
#                 # candidate['name'] = random.choice(settings['name'][race][gender])
#                 race = candidate['race']
#                 candidate['name'] = random.choice(settings['name'][race])
#             else:
#                 candidate[key] = random.choice(settings[key])
        
#     if 'gender' in candidate.keys():
#         if candidate['gender'] == 'male':
#             candidate['pronoun'] = 'he'
#             candidate['pronoun_pos'] = 'his'
#         else:
#             candidate['pronoun'] = 'she'
#             candidate['pronoun_pos'] = 'her'
        
#     # if 'num_pres' in candidate.keys():
#     #     candidate['num_pres'] = random.choice(np.arange(candidate['num_ecs']))
        
#     return candidate

def sample_one(settings, custom_stats=None):
    candidate = {}
    for key in settings:
        if custom_stats and key in custom_stats:
            candidate[key] = custom_stats[key]
        else:
            if key == 'letters_quality' or key == 'topic':
                char_index = candidate.get('character_index', 0)  # Default to 0 if not set
                candidate[key] = random.choice(settings[key][char_index])
            elif key == 'name':
                race = candidate.get('race', random.choice(list(settings['name'].keys())))  # Ensure race is set
                candidate['name'] = random.choice(list(settings['name'][race]))
            else:
                candidate[key] = random.choice(list(settings[key]))
    
    if 'gender' in candidate.keys():
        if candidate['gender'] == 'male':
            candidate['pronoun'] = 'he'
            candidate['pronoun_pos'] = 'his'
        else:
            candidate['pronoun'] = 'she'
            candidate['pronoun_pos'] = 'her'
    
    return candidate

"""
Get an applicant's race from a prompt
"""
def get_race(prompt):
    words = prompt.split(" ")
    for word in words:
        race = word.split(".")[0]
        if race in ["White", "Black", "Latino", "Asian"]:
            return race


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
    elif dataset == 'admissions_names':
        prompt = template.format(
            race = candidate['race'],
            name = candidate['name'],
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
    elif dataset == 'hiring_names':
        prompt = template.format(
            role = candidate['role'],
            race = candidate['race'],
            name = candidate['name'],
            exp = candidate['experience'],
            degree = candidate['degree'],
            referrals = candidate['referrals']
        )
    return prompt


"""
Tokenize the label. The returned token index is somewhat
arbitrary because it depends on the tokenizer.
"""
def format_label(label_eng, model_name):
    if model_name == "llama3":
        if label_eng == 'Yes':
            return 9642
        else:
            return 2822
    if model_name == "alpaca":
        if label_eng == 'Yes':
            return 8241
        else:
            return 3782
    elif model_name == "mistral":
        if label_eng == 'Yes':
            return 5613
        else:
            return 2501
    elif model_name == "gemma":
        if label_eng == 'Yes':
            return 3553
        else:
            return 1294


"""
Given an Intervenable object with BDAS intervention, 
return its rotation matrix and boundary masks
"""
def get_bdas_params(align_path, model_config):
    intervention_params = pv.BoundlessRotatedSpaceIntervention(
        embed_dim=model_config.hidden_size
    )
    intervention_params.load_state_dict(
        torch.load(align_path, weights_only=True)
    )
    
    rotate_layer = intervention_params.rotate_layer
    Q = rotate_layer.weight

    intervention_boundaries = intervention_params.intervention_boundaries
    intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)

    intervention_population = intervention_params.intervention_population
    temperature = intervention_params.temperature

    boundary_mask = sigmoid_boundary(
        intervention_population, 
        0.,
        intervention_boundaries[0] * model_config.hidden_size,
        temperature
    ).round()
    
    return intervention_params, Q, boundary_mask


"""
A wrapper to run inference on <model> with <tokenizer>.
Accepts a list of strings and returns a list of strings.
"""
def llm_predict(model, tokenizer, device, input_batch, 
                generate=False, gen_length=None):
    input_ids = tokenizer(input_batch, 
                          return_tensors="pt", 
                          padding=True).to(model.device)
    input_len = input_ids['input_ids'].shape[1]

    with torch.no_grad():
        if generate:
            output_ids = model.generate(**input_ids, 
                                        max_length=input_len+gen_length,
                                        # temperature=0.5,
                                        )
            output_preds = tokenizer.batch_decode(output_ids[:, input_len:], 
                                                  skip_special_tokens=False, 
                                                  clean_up_tokenization_spaces=False)
        else:
            output_batch = model(**input_ids)
            output_ids = output_batch['logits'][:, -1, :].argmax(dim=-1)
            output_preds = tokenizer.batch_decode(output_ids, 
            skip_special_tokens=True)
            
    return output_preds


"""
Saving a trained alignment.
"""
def save_alignment(intervenable, save_path, save_name):
    key = list(intervenable.interventions.keys())[0]
    intervention_params = intervenable.interventions[key][0]
    params_save_path = os.path.join(save_path, 
                                    save_name + '/model_params.pt')
    os.makedirs(os.path.dirname(params_save_path), exist_ok=True)
    torch.save(intervention_params.state_dict(), params_save_path)


"""
Load a trained alignment. Assumes the user is only
loading in one alignment potentially across multiple layers.
"""
def load_alignment(save_path, config, model, src_save_path=None,
                   alignment_type=pv.BoundlessRotatedSpaceIntervention,
                   interchange_dim=None):
    intervenable = pv.IntervenableModel(config, model)

    if save_path:
        # We assume the model is saved with this file name
        model_params_path = os.path.join(save_path, "model_params.pt")
        intervention_params = alignment_type(
            embed_dim=model.config.hidden_size
        )
        intervention_params.load_state_dict(
            torch.load(model_params_path, weights_only=True)
        )

        if src_save_path != None:
            src_params_path = os.path.join(src_save_path, "model_params.pt")
            src_intervention_params = alignment_type(
                embed_dim=model.config.hidden_size
            )
            src_intervention_params.load_state_dict(
                torch.load(src_params_path, weights_only=True)
            )
            src_rotate = src_intervention_params.rotate_layer
            intervention_params.set_src_rotate_layer(src_rotate)

        keys = list(intervenable.representations.keys())
        for key in keys:
            hook = intervenable.interventions[key][1]
            intervenable.interventions[key] = (intervention_params, hook)

    else:
        if interchange_dim != None:
            keys = list(intervenable.representations.keys())
            for key in keys:
                intervenable.interventions[key][0].interchange_dim = torch.tensor(interchange_dim)
    
    return intervenable

def get_race_pos(prompt):
    words = prompt.split()
    for i, word in enumerate(words):
        if word.split(".")[0] in ["White", "Black", "Latino", "Asian"]:
            race_idx = i - len(words)
    return race_idx

# def get_race(prompt, pos):
#     return prompt.split()[pos]

def color_race(race):
    if 'White' in race:
        return 'red'
    elif 'Black' in race:
        return 'blue'
    elif 'Latino' in race:
        return 'purple'
    elif 'Asian' in race:
        return 'green'
    

