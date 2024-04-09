cd src

experiment_folder="/hdd/hjl8708/experiments"

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/setting_test_reclor_AMR-LDA.json" \
    --output_path "../result/Mixtral_instruct-AMR-LDA_4shot" \
    --model_weight_dir None \
    --bsz 2 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name 'mixtral-8x7b-instruct' \
    --pretrained_model_name_or_path 'mistralai/Mixtral-8x7B-Instruct-v0.1' \
    --fewshot_samples_folder '/hdd/hjl8708/workspace/MCQA_option_finetuning/data/fewshots/ReClor_AMR-LDA_4shot'

