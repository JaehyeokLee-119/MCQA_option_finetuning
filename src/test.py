import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_HOME"] = "/hdd/hjl8708/saved_models"
os.environ["TRANSFORMERS_CACHE"] = "/hdd/hjl8708/saved_models"

from expl_generation import predict_answer
import fire 

# python main.py

BASE_PATH = '/hdd/hjl8708/workspace/NewCode'

def start(
    test_data_path=f'{BASE_PATH}/data/test_data/setting_test_reclor.json', # "../data/setting_EM_test.json"
    exp_dir=f'/hdd/hjl8708/experiments/mixtral-8x7b-nochat-ep2-setting1_LoRA_text_with_label',
    bsz=4, # bsz=8
    model_name='mixtral-instruct-8x7b',#'llama2-chat-13B',
    pretrained_model_name_or_path='mistralai/Mixtral-8x7B-Instruct-v0.1',#'meta-llama/Llama-2-13b-chat-hf',
    use_lora=True,
    model_precision="bf16", # 'bf16'
):
    print(f'use_lora: {use_lora} (type: {type(use_lora)})')
    
    predict_answer(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                     bsz=bsz, load_model_weights_dir=exp_dir, data_fname=test_data_path,
                     out_dir=exp_dir, use_lora=use_lora, max_new_tokens=50, model_precision=model_precision)
    
    

if __name__ == '__main__':
    fire.Fire(start)