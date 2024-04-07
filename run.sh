cd src

experiment_folder="/hdd/hjl8708/experiments"
lilist=("1" "2" "4")
for i in "${lilist[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --train_val_data_path "../data/train_data/setting${i}_2.json" \
        --exp_dir "${experiment_folder}/llama_nochat_ep2-answer_label-setting${i}_LoRA_text_with_label" \
        --num_epochs 2 \
        --bsz 16 \
        --patience 0 \
        --model_name "llama2-13B" \
        --use_lora True \
        --use_wandb True \
        --pretrained_model_name_or_path "meta-llama/Llama-2-13b-hf" \
        --project_name "Llama-nochat-setting_${i}_2_text_with_label_LoRA"

    CUDA_VISIBLE_DEVICES=0,1 python test.py \
        --test_data_path "../data/test_data/setting_test_reclor.json" \
        --exp_dir "${experiment_folder}/llama_nochat_ep2-answer_label-setting${i}_LoRA_text_with_label" \
        --bsz 32 \
        --model_name "llama2-13B" \
        --pretrained_model_name_or_path "meta-llama/Llama-2-13b-hf" \
        --use_lora True
done