import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["HF_HOME"] = "/hdd/hjl8708/saved_models"
os.environ["TRANSFORMERS_CACHE"] = "/hdd/hjl8708/saved_models"

from expl_generation import train_hyperparam_tuning
import fire 
import torch

# python main.py

BASE_PATH = '/hdd/hjl8708/workspace/NewCode'

def start(
    train_val_data_path=f'{BASE_PATH}/data/train_data/setting4_2.json',
    exp_dir=f'/hdd/hjl8708/experiments/TESTTEST',
    qn_type='yn',
    expl_type='cot',
    lrs=[1e-4], # [1e-5, 3e-5, 1e-4]
    num_epochs=2,
    bsz=16, # bsz=8
    patience=0,
    model_name='llama2-chat-13B',#'mixtral-8x7b',#'llama2-chat-13B',
    pretrained_model_name_or_path='meta-llama/Llama-2-13b-chat-hf',#'mistralai/Mixtral-8x7B-Instruct-v0.1',#'mistralai/Mixtral-8x7B-v0.1',#'meta-llama/Llama-2-13b-chat-hf',
    use_lora=True,
    use_wandb=True,
    project_name="Yes-instruction Setting4 LoRA sentence_and_label Mixtral8x7B (1e-4)",
    model_precision="bf16", # 'bf16'
):
    wandb_config = {"project": project_name, "entity": "jaehyeok-119"}
    
    # passage, question, label, option_sentence
        
    train_hyperparam_tuning(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                            data_fname=train_val_data_path, qn_type=qn_type, expl_type=expl_type, 
                            lrs=lrs, num_epochs=num_epochs, bsz=bsz, patience=patience, effective_bsz=32,
                            exp_dir=exp_dir, use_lora=use_lora, use_wandb=use_wandb, wandb_config=wandb_config, model_precision=model_precision)
    
if __name__ == '__main__':
    fire.Fire(start)
    