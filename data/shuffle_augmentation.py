# Type 1: 순서 바꾸기
# -> 각 문제마다 answers를 랜덤하게 섞은 버전 3개를 만든다

import os 
import pandas as pd 
import random 
from tqdm import tqdm 
# 현재 폴더의 모든 폴더에서 .jsonl 파일을 찾아서 data_paths에 저장
# data_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk('MMLU') for f in filenames if os.path.splitext(f)[1] == '.jsonl']
# data_paths += [os.path.join(dp, f) for dp, dn, filenames in os.walk('MedMCQA') for f in filenames if os.path.splitext(f)[1] == '.jsonl']
data_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk('LogiQA-2') for f in filenames if os.path.splitext(f)[1] == '.jsonl']
# data_paths += [os.path.join(dp, f) for dp, dn, filenames in os.walk('ReClor') for f in filenames if os.path.splitext(f)[1] == '.jsonl']
# data_paths += [os.path.join(dp, f) for dp, dn, filenames in os.walk('RULE') for f in filenames if os.path.splitext(f)[1] == '.jsonl']

# option이 들은 거 빼기
data_paths = [x for x in data_paths if 'options' not in x]
copies = 3

for data_path in data_paths:
    output_datapath = data_path.split('.')[0] + '_shuffled.jsonl'

    random.seed(42)

    data = pd.read_json(data_path, lines=True)
    # original_answer 는 answers 속에서 label에 해당하는 답을 뽑아낸 것
    data['original_answer'] = data.apply(lambda x: x['answers'][int(ord(x['label']))-ord('A')], axis=1)
    shuffled_data = data.copy()
    # shuffled_data의 모든 row를 삭제
    shuffled_data = shuffled_data[0:0]


    # 모든 data row에 대해서
    for i in tqdm(range(len(data)), desc=f'Processing {data_path}'):
        for j in range(copies):
            # data의 i번째 row를 복사해서 새로운 row를 만들고
            new_row = data.iloc[i].copy()
                # 같은 id_string을 갖는 shuffled_data 속 문제들과, new_row의 label이 겹치지 않아야 함
            # while new_row['answers'].index(new_row['original_answer']) == ord(new_row['label']) - ord('A'):
            # new_row의 answers를 섞어서 저장 (원래 답이랑 정답의 위치(original_answer)가 같으면 안됨)
            new_row['answers'] = random.sample(new_row['answers'], len(new_row['answers']))
            # new_row를 shuffled_data에 concat로 row 추가 (pandas 2.0이라서 append는 없다)
            shuffled_data = pd.concat([shuffled_data, new_row.to_frame().T])
                    
    shuffled_data['label'] = shuffled_data.apply(lambda x: chr(ord('A') + x['answers'].index(x['original_answer'])), axis=1)

    shuffled_data['id_string_original'] = shuffled_data['id_string']


    # _shuffled를 붙이되, 서로 겹치지 않게 숫자를 추가함
    shuffled_data['id_string'] = shuffled_data['id_string'] + '_shuffled'
    shuffled_data['id_string'] = shuffled_data.groupby('id_string').cumcount().add(1).astype(str).radd('_').radd(shuffled_data['id_string'])

    shuffled_data.to_json(output_datapath, orient='records', lines=True)
    print(f'{output_datapath} saved!')