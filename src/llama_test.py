import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["HF_HOME"] = "/hdd/hjl8708/saved_models"
os.environ["TRANSFORMERS_CACHE"] = "/hdd/hjl8708/saved_models"

from expl_generation import predict_answer
import fire 

# python main.py

BASE_PATH = '/hdd/hjl8708/workspace/explanation-consistency-finetuning'
        
def start(
    test_data_path=f'/hdd/hjl8708/workspace/explanation-consistency-finetuning/data/test_data/setting_test_reclor.json',
    exp_dir=f"/hdd/hjl8708/experiments/Test-4shotMixtral_instruct",
    model_dir=f'/hdd/hjl8708/saved_models/EC-finetuning',
    qn_type='yn',
    expl_type='cot',
    lrs=[1e-5, 3e-5, 1e-4],
    num_epochs=10,
    bsz=32,
    patience=0,
    model_name='mixtral-8x7b-instruct',#'llama-13b',#'mixtral-8x7b-instruct',#,'mixtral-8x7b',#'mixtral-8x7b-instruct',#,
    pretrained_model_name_or_path='mistralai/Mixtral-8x7B-Instruct-v0.1',#'meta-llama/Llama-2-13b-hf',#'mistralai/Mixtral-8x7B-Instruct-v0.1',#'mistralai/Mixtral-8x7B-v0.1',#
    project_name="Setting4 LoRA sentence_and_label Test (1e-4)",
    model_precision="bf16", # 'bf16'\
    fewshot_samples_folder=None,
    # fewshot_samples_folder='/hdd/hjl8708/workspace/explanation-consistency-finetuning/data/fewshots/ReClor_4shot'# None,
):
    print(f"fewshot_samples_folder: {fewshot_samples_folder}")
    
    model_dir = f'{exp_dir}/model'
    
    predict_answer(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                     load_model_weights_dir=None, data_fname=test_data_path, qn_type=qn_type, expl_type=expl_type,
                     out_dir=model_dir, use_lora=False, max_new_tokens=9, bsz=bsz, model_precision=model_precision, fewshot_samples_folder=fewshot_samples_folder)

if __name__ == '__main__':
    fire.Fire(start)