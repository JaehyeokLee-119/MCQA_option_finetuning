import os 
import pandas as pd 
import random
from tqdm import tqdm

data_paths = ['RULE_mainq_AMR-LDA.jsonl']

for data_path in data_paths:
    output_datapath = data_path.split('.')[0] + '_options.jsonl'
    data = pd.read_json(data_path, lines=True)

    option_data = data.copy()
    option_data = option_data[0:0]

    for i in tqdm(range(len(data)), desc=f'Processing {data_path}'):
        for j in range(4):
            new_row = data.iloc[i].copy()
            new_row['target_option_alphabet'] = chr(ord('A') + j)
            new_row['target_option_string'] = new_row['answers'][j]
            
            new_row['target_option_correctness'] = True if ord(new_row['label']) - ord('A') == j else False
            
            new_row['id_string_original'] = new_row['id_string']
            new_row['id_string'] = new_row['id_string'] + '_option' + chr(ord('A') + j)
            option_data = pd.concat([option_data, new_row.to_frame().T])
    
    option_data.to_json(output_datapath, orient='records', lines=True)#, indent=4)
    print(f'{output_datapath} saved!')