from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup, LlamaForCausalLM, LlamaTokenizer
import transformers
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
import os
import numpy as np
import dotenv
dotenv.load_dotenv()
API_KEY = os.getenv('HUGGINGFACE_AUTH_TOKEN')
from peft import (
    LoraConfig,
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PeftModel,
)
from peft.peft_model import PeftModelForCausalLM
from peft.utils import _prepare_prompt_learning_config


import wandb


def get_peft_model(model, peft_config, adapter_name: str = "default"):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name)
    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)

class CLM_wrapper:    
    def __init__(self, model_name, pretrained_model_name_or_path, load_model_weight_dir=None, model_parallel=False, use_lora=False, model_precision='bf16', hf_home=None):
        device_map = None if model_parallel is False else 'auto'
        print(f'load_model_weight_dir: {load_model_weight_dir}')
        
        def load_model_with_precision(pretrained_model_name_or_path, device_map, model_precision):
            if model_precision == 'bf16':
                return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map, token=API_KEY, cache_dir=hf_home)
            elif model_precision == 'fp16':
                return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, device_map=device_map, token=API_KEY, cache_dir=hf_home)
            elif model_precision == '8-bit':
                if pretrained_model_name_or_path.startswith('mistralai'):
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, load_in_8bit=True, device_map=device_map, token=API_KEY, cache_dir=hf_home)
                else:
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.int8, device_map=device_map, token=API_KEY, cache_dir=hf_home)
            elif model_precision == '4-bit':
                if pretrained_model_name_or_path.startswith('mistralai'):
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, load_in_4bit=True, device_map=device_map, token=API_KEY, cache_dir=hf_home)
                else:
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32, device_map=device_map, token=API_KEY, cache_dir=hf_home)
            else:
                raise ValueError(f"Invalid model_precision: {model_precision}")
                    
        if use_lora == True:
            if load_model_weight_dir is None: # 
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, token=API_KEY)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'left'
                # setup model
                self.model = load_model_with_precision(pretrained_model_name_or_path, device_map, model_precision)
                # AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map, token=API_KEY)
                # llama_model = load_model_with_precision('meta-llama/Llama-2-13b-chat-hf', device_map, 'bf16')
                
                lora_r = 16
                if pretrained_model_name_or_path.startswith('meta-llama'):
                    lora_alpha = 16
                    lora_target_modules = ["gate_proj", "down_proj", "up_proj"]
                elif pretrained_model_name_or_path.startswith('mistralai'):
                    lora_alpha = 8
                    lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
                    # lora_target_modules = ['w1', 'w2', 'w3']
                    
                lora_dropout: float = 0.05
                
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM")
                
                self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, config)
                self.model.print_trainable_parameters()
                
                if device_map is None:
                    self.model.cuda()
                    
            else:  # if load_model_weight_dir is Not None:
                # from safetensors.torch import load_file
                # load_model_weight_dir에 있는 checkpoint라는 이름이 포함된 하위하위 folder name 찾기
                for root, dirs, files in os.walk(load_model_weight_dir):
                    for dir in dirs:
                        if "checkpoint" in dir: # full path
                            lora_load_model_weight_dir = os.path.join(root, dir)
                            break
                        else:
                            pass
                
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, token=API_KEY)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'left'
                base_model = load_model_with_precision(pretrained_model_name_or_path, device_map, model_precision)
                # base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map, token=API_KEY)
                self.model = PeftModel.from_pretrained(base_model, lora_load_model_weight_dir, device_map=device_map)
                print("PEFT loaded")
                
        else: # use_Lora == False
            load_model_weight_dir = f'{load_model_weight_dir}/model/model.pkl'
            # setup tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, token=API_KEY) #use_auth_token=API_KEY
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            # setup model
            self.model = load_model_with_precision(pretrained_model_name_or_path, device_map, model_precision)
            
            if device_map is None:
                self.model.cuda()
            if (load_model_weight_dir is not None) and ('None' not in load_model_weight_dir):
                self.model.load_state_dict(torch.load(load_model_weight_dir, map_location='cpu')) # 'cpu'
        self.model_name = model_name

    def collate_fn(self, dataset):
        # dataset is a list of (input, output), input=str, output=str
        # sanity check
        for ex_idx in range(len(dataset)):
            input, output = dataset[ex_idx]
            # assert '\n' not in input and '\n' not in output
            assert '\n' not in output
            if self.model_name.startswith('gpt'):
                dataset[ex_idx] = (input.strip(), ' ' + output.strip())
            else:
                dataset[ex_idx] = (input.strip() + ' ', output.strip())

        input_output_texts = [ex[0] + ex[1] for ex in dataset] # 학습할 string의 list
        input_dict = self.tokenizer(input_output_texts, padding=True, return_tensors="pt")
        num_examples = len(input_output_texts)

        # "build output labels" (pad tokens and input tokens are masked with 100 in the labels variable)
        labels = input_dict['input_ids'].detach().clone()
        for example_idx in range(num_examples):
            # 각 dataset에 대해서 output에 해당하는 부분을 Tokenizing해서 뽑아냄 
            encoded_output = self.tokenizer(dataset[example_idx][1])['input_ids']
            
            if self.model_name.startswith('alpaca') or self.model_name.startswith('llama2') or self.model_name.startswith('mixtral'):
                assert encoded_output[0] == self.tokenizer.bos_token_id
                # 'alpaca' 혹은 'llama2' 모델이면 다음과 같이 추가 처리
                encoded_output = encoded_output[1:]
            
            # 이 과정이 왜 필요한가? 
            assert labels[example_idx][-len(encoded_output):].tolist() == encoded_output
            
            # labels의 output에 해당하는 부분을 -100으로 채움
            labels[example_idx][:-len(encoded_output)] = -100
        input_dict['labels'] = labels
        return input_dict

    def collate_fn_nolabel(self, inputs):
        # inputs is a list of input (str)
        inputs = [x.strip() for x in inputs]
        input_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")
        return input_dict

    def compute_dev_loss(self, model, dev_dataset, epoch_num):
        """검증 데이터셋에 대한 손실 계산 및 로깅"""
        model.eval()  # 모델을 평가 모드로 설정
        dev_loss = []
        for batch in tqdm(dev_dataset, desc=f'Epoch {epoch_num} dev loss'):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                dev_loss.append(loss.item())

        mean_dev_loss = np.mean(dev_loss)
        print(f"Initial Dev Loss: {mean_dev_loss:.4f}")
        # wandb.log({"dev_loss": mean_dev_loss})
        return mean_dev_loss

    def train_LoRA(self, train_data, dev_data, lr, num_epochs, bsz, num_grad_acc, patience, output_dir,
                   shuffle=True, use_lr_scheduler=False, use_wandb=False, wandb_config=None):
        
        if use_wandb:
            wandb.login()
            wandb.init(**wandb_config)
            wandb_run_name = wandb_config['project']
            
        assert not os.path.exists(output_dir), output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.model.gradient_checkpointing_enable()
        
        with open(f'{output_dir}/train.log', 'a') as f:
            f.write(str({'lr': lr, 'num_epochs': num_epochs, 'bsz': bsz, 'num_grad_acc': num_grad_acc, 'patience': patience,
                         'shuffle': shuffle, 'use_lr_scheduler': use_lr_scheduler}) + '\n')
        # load model
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
        dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=bsz, collate_fn=self.collate_fn)
        
        # Dev loss first (before)
        epoch_start_time = time.time()
        dev_loss = self.compute_dev_loss(self.model, dev_dataloader, -1)
        epoch_end_time = time.time()
        with open(f'{output_dir}/train.log', 'a') as f:
            f.write(
                f'Before train dev loss {dev_loss:.3f} ({(epoch_end_time - epoch_start_time)//60} min)\n')
            wandb.log({"eval/loss": dev_loss})
            
        self.model.train()
        training_args = transformers.TrainingArguments(
            output_dir=output_dir,  # 학습된 모델과 체크포인트를 저장할 디렉토리
            num_train_epochs=num_epochs, #num_epochs,  # 학습 에포크 수
            per_device_train_batch_size=bsz,  # 각 디바이스 별 학습 배치 크기
            per_device_eval_batch_size=bsz,  # 각 디바이스 별 평가 배치 크기
            gradient_accumulation_steps=num_grad_acc,  # 그래디언트 축적 스텝 수
            evaluation_strategy="epoch",  # 에포크마다 평가
            save_strategy="epoch",  # 에포크마다 모델 저장
            optim='adamw_torch',
            load_best_model_at_end=True,  # 학습 종료 시 최적 모델을 로드
            metric_for_best_model="eval_loss",  # 최적 모델 선정 기준
            greater_is_better=False,  # 손실 감소가 성능 향상을 의미
            warmup_steps=100,  # 러닝 레이트 웜업을 위한 스텝 수
            learning_rate=lr,  # 러닝 레이트
            report_to="wandb" if use_wandb else None,  # Weights & Biases 로깅 활성화
            logging_dir='./logs',  # 로깅 디렉토리 설정 (W&B 사용 시)
            logging_steps=5,  # 로깅 간격
            save_total_limit=1,  # 저장할 최대 체크포인트 수
            run_name=wandb_run_name,
        )
        
        # training (with transformer.Trainer)
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_data, 
            eval_dataset=dev_data,
            args=training_args,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=1)],
        )
        trainer.train()
        
        # dev loss
        dev_loss = self.compute_dev_loss(self.model, dev_dataloader, -1)
        return dev_loss

    def train(self, train_data, dev_data, lr, num_epochs, bsz, num_grad_acc, patience, output_dir, 
              shuffle=True, use_lr_scheduler=False, use_wandb=False, wandb_config=None):
        
        if type(self.model) == PeftModelForCausalLM:
            print("Do Train_LoRA")
            return self.train_LoRA(train_data, dev_data, lr, num_epochs, bsz, num_grad_acc, patience, output_dir,
                            shuffle, use_lr_scheduler, use_wandb, wandb_config)
        
        if use_wandb:
            wandb.login()
            wandb.init(**wandb_config)
            wandb_run_name = wandb_config['project']
        
        assert not os.path.exists(output_dir), output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.model.gradient_checkpointing_enable()

        with open(f'{output_dir}/train.log', 'a') as f:
            f.write(str({'lr': lr, 'num_epochs': num_epochs, 'bsz': bsz, 'num_grad_acc': num_grad_acc, 'patience': patience,
                         'shuffle': shuffle, 'use_lr_scheduler': use_lr_scheduler}) + '\n')
            
        # load model
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # training
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        optimizer.zero_grad()
        optimal_dev_loss = np.inf
        optimal_epoch = 0
        if use_lr_scheduler:
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, 
                                                           num_training_steps=len(train_data) // (bsz * num_grad_acc) * num_epochs)
        
        optimal_model = None
        
        if use_wandb:
            wandb.log({"dev_loss": dev_loss})
            
        for epoch in trange(num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = []
            
            # prepare data
            train_dataloader = DataLoader(train_data, 
                                        shuffle=shuffle, 
                                        batch_size=bsz, 
                                        collate_fn=self.collate_fn)
            # train
            batch_idx = 0
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}'):
                batch = {k: v.cuda() for k, v in batch.items()}
                if 'token_type_ids' in batch:
                    del batch['token_type_ids']
                loss = self.model(**batch).loss / num_grad_acc
                loss.backward()
                train_loss.append(loss.item() * num_grad_acc)
                
                if (batch_idx + 1) % num_grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if use_lr_scheduler:
                        lr_scheduler.step()

                    # wandb logging
                    if use_wandb:
                        loss_accumulated = np.mean(train_loss[-num_grad_acc:])
                        wandb.log({"train_loss": loss_accumulated.item()})
                        
                batch_idx += 1
                
                
            train_loss = np.mean(train_loss)
            
            print(f"Start evaluating on dev data: (epoch {epoch}/{num_epochs})")
            
            # evaluate on dev
            dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=bsz, collate_fn=self.collate_fn)
            epoch_start_time = time.time()
            dev_loss = self.compute_dev_loss(self.model, dev_dataloader, epoch)
            epoch_end_time = time.time()
            with open(f'{output_dir}/train.log', 'a') as f:
                f.write(
                    f'Epoch {epoch} Train dev loss: {dev_loss:.3f} ({(epoch_end_time - epoch_start_time)//60} min)\n')
            if use_wandb:
                wandb.log({"dev_loss": dev_loss})

            if dev_loss < optimal_dev_loss:
                print(f"[New optimal: epoch {optimal_epoch}->{epoch}] Current dev_loss: {dev_loss:.4f}({epoch}), previous optimal loss : {optimal_dev_loss:.4f} ({optimal_epoch})")
                optimal_epoch = epoch
                optimal_dev_loss = dev_loss
                optimal_model = deepcopy(self.model.state_dict())
            else:
                print(f"[Keep optimal: epoch {optimal_epoch}] Current dev_loss: {dev_loss:.4f}({epoch}), optimal loss : {optimal_dev_loss:.4f} ({optimal_epoch})")
            
            epoch_end_time = time.time()
            with open(f'{output_dir}/train.log', 'a') as f:
                f.write(
                    f'Epoch {epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f} ({(epoch_end_time - epoch_start_time)//60} min)\n')

            if epoch - optimal_epoch > patience: # 모델을 저장하는 경우
                print(f"End training at {optimal_epoch}. saving optimal model")
                
                start_time = time.time()
                torch.save(optimal_model, f'{output_dir}/model.pkl') #
                end_time = time.time()
                time_consumed_string = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
                print(f"End saving model. Time consumed: {time_consumed_string}")
                
                del self.model
                return optimal_dev_loss
    
        print(f"End training at {epoch}. saving last model")
        start_time = time.time()
        torch.save(optimal_model, f'{output_dir}/model.pkl')
        end_time = time.time()
        time_consumed_string = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        
        print(f"End saving model. Time consumed: {time_consumed_string}") # 시간을 형식에 맞춰서 
        return optimal_dev_loss


    def predict(self, inputs, bsz, eos_token_id, do_sample, num_beams=None, top_p=None, num_return_sequences=1, max_new_tokens=600):
        if do_sample:
            assert (top_p is not None) and (num_beams is None)
        else:
            assert (num_beams is not None) and (top_p is None)
        if num_return_sequences > 1:
            assert do_sample
            
        self.model.eval()
        outputs = []
        
        dataloader = DataLoader(inputs, batch_size=bsz, collate_fn=self.collate_fn_nolabel)
        for batch in tqdm(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            if 'token_type_ids' in batch:
                del batch['token_type_ids']
            with torch.no_grad():
                if do_sample: # sampling
                    if self.model_name.startswith('mixtral'):
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, num_return_sequences=num_return_sequences,
                                                        early_stopping=True, eos_token_id=eos_token_id, pad_token_id=2)
                    else:
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, num_return_sequences=num_return_sequences,
                                                        early_stopping=True, eos_token_id=eos_token_id)
                    
                else: # greedy/beam search
                    if self.model_name.startswith('mixtral'):
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                                        early_stopping=True, eos_token_id=eos_token_id, pad_token_id=2)
                    else:
                        input_outputs = self.model.generate(**batch, do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                                        early_stopping=True, eos_token_id=eos_token_id)
            assert len(input_outputs) == len(batch['input_ids']) * num_return_sequences
            # duplicate batch['input_ids'] by num_return_sequences number of times (to match input_outputs)
            batch_input_ids = torch.stack([batch['input_ids'][ex_idx] for ex_idx in range(len(batch['input_ids'])) for _ in range(num_return_sequences)])
            assert len(batch_input_ids) == len(input_outputs)
            assert torch.all(input_outputs[:, : batch_input_ids.shape[1]] == batch_input_ids)
            batch_outputs = input_outputs[:, batch_input_ids.shape[1]:].cpu().tolist()
            for output in batch_outputs:
                if eos_token_id not in output:
                    outputs.append(output)
                else:
                    outputs.append(output[: output.index(eos_token_id)])
            
        output_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert len(output_texts) == len(inputs) * num_return_sequences
        output_texts = [text.strip() for text in output_texts]
        if num_return_sequences == 1:
            return output_texts
        else:
            assert len(output_texts) % num_return_sequences == 0
            output_texts = [output_texts[pos: pos + num_return_sequences] for pos in range(0, len(output_texts), num_return_sequences)]
            assert len(output_texts) == len(inputs)
            return output_texts