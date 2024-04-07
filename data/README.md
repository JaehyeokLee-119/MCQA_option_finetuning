## prompt augmentation processing
1. generate Augmented result with AMR-LDA_prompt_augmentation
2. run augment_reflecting.py to reflect augmented result to answers column in the jsonl file
3. run option_augmentation.py to generate option problems 
4. run data_forming.ipynb to aggregate original problems + option problems then make ec-finetuning format data file