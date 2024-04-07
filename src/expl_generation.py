from llama2_utils import convert_llama2_prompt_format
import string
import json
import numpy as np
from model_wrapper import CLM_wrapper
from transformers import AutoTokenizer
import os
import shutil

DEBUG = False

def embed_prompt_with_fewshot(qn_type, chat_prompt, expl_type, question, answer=None, explanation=None, eos_token=None, test=False, fewshot_samples_folder=None):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    ''
    # answer가 'A', 'B', 'C', 'D' 중 하나인 경우
    punc_removed_answer = remove_punc(answer)
    if punc_removed_answer in ['yes', 'no']:
        qn_type = 'yn'
    else: 
        qn_type = 'mc'
    
    if fewshot_samples_folder is not None:
        if qn_type == 'yn':
            samples = json.load(open(f'{fewshot_samples_folder}/Option_fewshot.json'))
        elif qn_type == 'mc': 
            samples = json.load(open(f'{fewshot_samples_folder}/MCQA_fewshot.json'))
            
    fewshot_prefix = ''
    if samples is not None:
        for sample in samples:
            sample_input, sample_output = embed_prompt(qn_type, chat_prompt, expl_type, sample['question'], sample['answer'], sentence=sample['answer_sentence'], eos_token='\n\n\n')
            fewshot_prefix += f'{sample_input}{sample_output}'
        input = embed_prompt(qn_type, chat_prompt, expl_type, question, answer, test=test, eos_token=eos_token)
        return fewshot_prefix+input

# cot/posthoc prompts (for sentence + label)
def embed_prompt(qn_type, chat_prompt, expl_type, question, answer=None, explanation=None, sentence=None, eos_token=None, test=False):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    punc_removed_answer = remove_punc(answer)
    if punc_removed_answer in ['yes', 'no']:
        qn_type = 'yn'
        input_instruct = 'Choose one of the following options (yes/no).'
    else: 
        qn_type = 'mc'
        input_instruct = 'Write the answer sentence and label (A/B/C/D).'
    
    if not chat_prompt: 
        input_str = f"{question} {input_instruct}"
        # input_str = input_str
        # output_str = label_first_part + label_answer_part
        label_first_part = "<Answer> The answer is: "
        if qn_type == 'mc':
            label_answer_part = f"{sentence} {answer}"
        elif qn_type == 'yn':
            if punc_removed_answer == 'yes':
                label_answer_part = f'{answer} "{sentence}" is the correct answer to the question.'
            else:
                label_answer_part = f'{answer} "{sentence}" is not the correct answer to the question.'
    else:
        system_prompt = f'I will ask you a question. Answer to the question.'
        input_str = convert_llama2_prompt_format(system_prompt, question, delimiter=input_instruct)
        
        label_first_part = "The answer is: "
        if qn_type == 'mc':
            label_answer_part = f"{sentence} {answer}"
        elif qn_type == 'yn':
            if punc_removed_answer == 'yes':
                label_answer_part = f'{answer} "{sentence}" is the correct answer to the question.'
            else:
                label_answer_part = f'{answer} "{sentence}" is not the correct answer to the question.'
        
    if test is True:
        return input_str + label_first_part
    else:
        output_str = f"{label_first_part}{label_answer_part}{eos_token}"
        return input_str, output_str
    

def extract_answer_from_output(qn_type, expl_type, text):
    assert qn_type in ['mc', 'yn']
    options = {'mc': ['A', 'B', 'C', 'D'], 'yn': ['yes', 'no']}[qn_type]
    text = text.strip()
    if expl_type == 'cot':
        answer_is = [
            f'So the answer is {option}' in text for option in options]
        if sum(answer_is) != 1:
            answer = 'neither'
            explanation = text
        else:
            answer = options[answer_is.index(1)]
            explanation = text[:text.index(
                f'So the answer is {answer}')].strip()
    elif expl_type == 'posthoc':
        answer_is = [text.startswith(option) for option in options]
        if sum(answer_is) != 1:
            answer = 'neither'
        else:
            answer = options[answer_is.index(1)]
        if 'Explanation:' in text:
            explanation = text[text.index(
                'Explanation:') + len('Explanation:'):].strip()
        else:
            explanation = text.strip()
    else:
        raise NotImplementedError
    return {'answer': answer, 'explanation': explanation}


# train model
def train_model(model_name, pretrained_model_name_or_path,
                data, qn_type, expl_type,
                model_parallel, lr, num_epochs, bsz, num_grad_acc, patience, output_dir, shuffle=True, use_lr_scheduler=False,
                use_lora=False, use_wandb=False, wandb_config=None, model_precision='bf16', fewshot_samples_folder=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    eos_token = tokenizer.eos_token
    chat_prompt = ('llama' in model_name and 'chat' in model_name) or ('instruct' in model_name)
    print(f"Train chat_prompt: {chat_prompt}")
    
    assert (type(data) == list) == (type(qn_type) == list)
    if type(data) != list:  # single-task learning
        if fewshot_samples_folder is not None:
            train_data = [embed_prompt_with_fewshot(qn_type, chat_prompt, expl_type, ex['question'],
                                    ex['answer'], ex['answer_sentence'], eos_token, fewshot_samples_folder=fewshot_samples_folder) for ex in data['train']]
            dev_data = [embed_prompt_with_fewshot(qn_type, chat_prompt, expl_type, ex['question'],
                                    ex['answer'], ex['answer_sentence'], eos_token, fewshot_samples_folder=fewshot_samples_folder) for ex in data['valid']]
        else:
            train_data = [embed_prompt(qn_type, chat_prompt, expl_type, question=ex['question'],
                                    answer=ex['answer'], sentence=ex['answer_sentence'], eos_token=eos_token) for ex in data['train']]
            dev_data = [embed_prompt(qn_type, chat_prompt, expl_type, question=ex['question'],
                                    answer=ex['answer'], sentence=ex['answer_sentence'], eos_token=eos_token) for ex in data['valid']]
    else:  # multi-task learning
        assert len(data) == len(qn_type)
        if fewshot_samples_folder is not None:
            train_data = [embed_prompt_with_fewshot(qn_type, chat_prompt, expl_type, ex['question'],
                                    ex['answer'], ex['answer_sentence'], eos_token, fewshot_samples_folder=fewshot_samples_folder) for ex in data['train']]
            dev_data = [embed_prompt_with_fewshot(qn_type, chat_prompt, expl_type, ex['question'],
                                    ex['answer'], ex['answer_sentence'], eos_token, fewshot_samples_folder=fewshot_samples_folder) for ex in data['valid']]
        else:
            train_data = [embed_prompt(task_qn_type, chat_prompt, expl_type, ex['question'], ex['answer'], ex['answer_sentence'], eos_token)
                        for task_data, task_qn_type in zip(data, qn_type) for ex in task_data['train']]
            dev_data = [embed_prompt(task_qn_type, chat_prompt, expl_type, ex['question'], ex['answer'], ex['answer_sentence'], eos_token)
                        for task_data, task_qn_type in zip(data, qn_type) for ex in task_data['valid']]
    # train + eval
    clm_wrapper = CLM_wrapper(
        model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path, model_parallel=model_parallel, use_lora=use_lora, model_precision=model_precision)
    dev_loss = clm_wrapper.train(train_data=train_data, dev_data=dev_data,
                                 lr=lr, num_epochs=num_epochs, bsz=bsz, num_grad_acc=num_grad_acc, patience=patience, output_dir=output_dir,
                                 shuffle=shuffle, use_lr_scheduler=use_lr_scheduler, use_wandb=use_wandb, wandb_config=wandb_config)
    
    return dev_loss

def train_hyperparam_tuning(model_name, pretrained_model_name_or_path,
                            data_fname, lrs, num_epochs, bsz, patience, exp_dir,
                            effective_bsz=32, use_lora=False, use_wandb=False, wandb_config=None, qn_type='yn', expl_type='cot', model_precision='bf16'):
    # load data
    if type(data_fname) == str:
        data = json.load(open(data_fname))
    elif type(data_fname) == list:
        data = [json.load(open(fname)) for fname in data_fname]
        assert (type(qn_type) == list) and (len(data_fname) == len(qn_type))
    num_grad_acc = effective_bsz // bsz
    dev_losses = []
    
    print(f'batch_size: {bsz}, effective_bsz: {effective_bsz} ➜ num_grad_acc: {num_grad_acc}')
    
    # # for debugging
    if DEBUG == True: 
        debug_len = 16
        data['train'] = data['train'][:debug_len]
        data['valid'] = data['valid'][:debug_len]
    
    for lr in lrs:
        print(f"Training with lr={lr}...")
            
        dev_loss = train_model(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                               data=data, qn_type=qn_type, expl_type=expl_type,
                               model_parallel=True, lr=lr, num_epochs=num_epochs, bsz=bsz, num_grad_acc=num_grad_acc, patience=patience,
                               output_dir=f'{exp_dir}/lr{lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/', use_lr_scheduler=True,
                               use_lora=use_lora, use_wandb=use_wandb, wandb_config=wandb_config, model_precision=model_precision)
        dev_losses.append(dev_loss)
    
    if use_lora == False:
        # choose the learning rate with lowest dev loss, change the file directory without the hyperparamters name; remove the other files
        optimal_lr = lrs[np.argmin(dev_losses)]
        optimal_output_dir = f'{exp_dir}/lr{optimal_lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/'
        
        model_folder = f'{exp_dir}/model/'
        # model_folder 없으면 만들기
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)
            
        # f'{optimal_output_dir}model.pkl' -이동> '{model_folder}model.pkl'
        os.rename(f'{optimal_output_dir}model.pkl', f'{model_folder}model.pkl')
        
        optimal_output_dir_new = f'{exp_dir}/model/'
        # optimal_output_dir_new = f'{exp_dir}/lr{optimal_lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}-SELECTED/'
        # os.rename(optimal_output_dir, optimal_output_dir_new)
        # remove checkpoints of suboptimal hyperparameters
        os.mkdir(f'{exp_dir}/hyperparameter_tuning')
        for lr in lrs:
            if lr != optimal_lr:
                os.remove(
                    f'{exp_dir}/lr{lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/model.pkl')
                shutil.move(f'{exp_dir}/lr{lr}_numepochs{num_epochs}_bsz{bsz}_gradacc{num_grad_acc}_patience{patience}/',
                            f'{exp_dir}/hyperparameter_tuning')
                
    return dev_losses


def predict_answer(model_name, pretrained_model_name_or_path, load_model_weights_dir,
                     data_fname, out_dir=None,
                     model_parallel=True, bsz=16, do_sample=False, 
                     num_beams=1, top_p=None, num_return_sequences=1,
                     use_lora=False, max_new_tokens=600, qn_type='yn', expl_type='cot', model_precision='bf16', fewshot_samples_folder=None):
    # load data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.eos_token
    chat_prompt = ('llama' in model_name and 'chat' in model_name) or ('instruct' in model_name)
    print(f"Test chat_prompt: {chat_prompt}")
    
    
    if type(data_fname) == str: # 데이터의 경로가 str인 경우 (보통 여기서 실행됨)
        data = json.load(open(data_fname))
        data = data['test'] 
        if DEBUG == True:
            data = data[:64] #debug
    elif type(data_fname) == list:
        data = data_fname
    else:
        raise NotImplementedError
    
    if fewshot_samples_folder is not None:
        eval_inputs = [embed_prompt_with_fewshot(qn_type=qn_type, chat_prompt=chat_prompt, test=True,
                                expl_type=expl_type, answer=ex['answer'], question=ex['question'], eos_token=eos_token, fewshot_samples_folder=fewshot_samples_folder) for ex in data]
    else:
        eval_inputs = [embed_prompt(qn_type=qn_type, chat_prompt=chat_prompt, test=True,
                                expl_type=expl_type, answer=ex['answer'], question=ex['question'], eos_token=eos_token) for ex in data]
    
    # eval_inputs의 길이 statistics
    eval_inputs_lens = np.array([len(input1) for input1 in eval_inputs])
    print(f"Max input length: {np.max(eval_inputs_lens)}, Min input length: {np.min(eval_inputs_lens)}, Mean input length: {np.mean(eval_inputs_lens)}")
    
    
    if load_model_weights_dir is None:
        clm_wrapper = CLM_wrapper(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                                  model_parallel=model_parallel, use_lora=False, model_precision=model_precision)
    else:
        clm_wrapper = CLM_wrapper(model_name=model_name, pretrained_model_name_or_path=pretrained_model_name_or_path,
                              load_model_weight_dir=load_model_weights_dir, model_parallel=model_parallel, 
                              use_lora=use_lora, model_precision=model_precision)
    
    # << Length test >>: eval_inputs 중에서 가장 긴 것들을 bsz 개수만큼 뽑아서 터지는지 보기
    eval_inputs_long = eval_inputs.copy()
    eval_inputs_long.sort(key=lambda x: len(x), reverse=True)
    eval_inputs_long = eval_inputs_long[:bsz]
    
    # eval_inputs_long에 대해서 predict 시킨다
    try:
        print("test prediction!")
        long_test_outputs = clm_wrapper.predict(inputs=eval_inputs_long, eos_token_id=eos_token_id, bsz=bsz,
                                            do_sample=do_sample, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences,
                                            max_new_tokens=max_new_tokens)
        
        # 디버깅용 랜덤인풋
        # eval_inputs_random_bsz = eval_inputs.copy()
        # np.random.shuffle(eval_inputs_random_bsz)
        # eval_inputs_random_bsz = eval_inputs_random_bsz[:bsz]
        # random_test_outputs = clm_wrapper.predict(inputs=eval_inputs_random_bsz, eos_token_id=eos_token_id, bsz=bsz,
        #                                     do_sample=do_sample, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences,
        #                                     max_new_tokens=max_new_tokens)
        print("Passed long test")
    except:
        print(f"Error occurred! too long (not enough GPU memory)")
        return None
        
    output_texts = clm_wrapper.predict(inputs=eval_inputs, eos_token_id=eos_token_id, bsz=bsz,
                                       do_sample=do_sample, num_beams=num_beams, top_p=top_p, num_return_sequences=num_return_sequences,
                                       max_new_tokens=max_new_tokens)
    if num_return_sequences == 1:
        preds = [extract_answer_from_output(
            qn_type, expl_type, text) for text in output_texts]
    else:
        preds = [[extract_answer_from_output(
            qn_type, expl_type, text) for text in ex_texts] for ex_texts in output_texts]
    
    assert len(preds) == len(data)
    
    for i in range(len(preds)):
        data[i]['output'] = output_texts[i]
        data[i].pop('explanation', None)
    
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if do_sample is False:
            assert num_beams == 1
            json.dump(data, open(
                f'{out_dir}/greedy_preds.json', 'w'), indent=4)
        else:
            json.dump(data, open(
                f'{out_dir}/sampling_preds_topp{top_p}.json'), indent=4)
    return data