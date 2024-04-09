cd src

experiment_folder="/hdd/hjl8708/experiments"

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/setting_test_reclor_AMR-LDA.json" \
    --output_path "../result/Llama-AMR-LDA_RULE_mainq-AMR4shot" \
    --model_weight_dir None \
    --bsz 4 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name "llama2-13B" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-13b-hf" \
    --fewshot_samples_folder '/hdd/hjl8708/workspace/MCQA_option_finetuning/data/fewshots/ReClor_AMR-LDA_4shot'

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/RULE_subq_all_AMR-LDA_augmented.json" \
    --output_path "../result/Llama-AMR-LDA_RULE_subq-AMR4shot" \
    --model_weight_dir None \
    --bsz 4 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name "llama2-13B" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-13b-hf" \
    --fewshot_samples_folder '/hdd/hjl8708/workspace/MCQA_option_finetuning/data/fewshots/ReClor_AMR-LDA_4shot'

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/setting_test_reclor_AMR-LDA.json" \
    --output_path "../result/Llama-chat-AMR-LDA_RULE_mainq" \
    --model_weight_dir None \
    --bsz 16 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name "llama2-chat-13B" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-13b-chat-hf"

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/RULE_subq_all_AMR-LDA_augmented.json" \
    --output_path "../result/Llama-chat-AMR-LDA_RULE_subq" \
    --model_weight_dir None \
    --bsz 16 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name "llama2-chat-13B" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-13b-chat-hf"

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/setting_test_reclor_AMR-LDA.json" \
    --output_path "../result/Llama-chat-AMR-LDA_RULE_mainq-AMR4shot" \
    --model_weight_dir None \
    --bsz 4 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name "llama2-chat-13B" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-13b-chat-hf" \
    --fewshot_samples_folder '/hdd/hjl8708/workspace/MCQA_option_finetuning/data/fewshots/ReClor_AMR-LDA_4shot'

CUDA_VISIBLE_DEVICES=0,1 python test.py \
    --test_data_path "../data/test_data/RULE_subq_all_AMR-LDA_augmented.json" \
    --output_path "../result/Llama-chat-AMR-LDA_RULE_subq-AMR4shot" \
    --model_weight_dir None \
    --bsz 4 \
    --use_lora false \
    --max_new_tokens 9 \
    --model_name "llama2-chat-13B" \
    --pretrained_model_name_or_path "meta-llama/Llama-2-13b-chat-hf" \
    --fewshot_samples_folder '/hdd/hjl8708/workspace/MCQA_option_finetuning/data/fewshots/ReClor_AMR-LDA_4shot'

    # 'llama-13B-chat'

    # --model_name 'mixtral-8x7b-instruct' \
    # --pretrained_model_name_or_path 'mistralai/Mixtral-8x7B-Instruct-v0.1' \