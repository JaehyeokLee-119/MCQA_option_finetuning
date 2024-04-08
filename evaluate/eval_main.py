import itertools
import json
from utils.tools import read_jsonl, check_jsonls, metric_max_over_ground_truths,  f1_score, exact_match_score, metric_match
import fire

def dtype_processing(pred):
    dtypes = ['logiqa', 'ReClor_val', 'RULE_train', 'MMLU_val']
    sub_types = ['sub', 'option', 'shuffle']
    
    result = ""
    qid = pred["qid"]
    
    for dtype in dtypes:
        result += dtype if dtype in qid else ""
    for sub_type in sub_types:
        result += sub_type if sub_type in qid else ""
    
    if 'option' in result:
        result += 'selective' if 'yes' in pred['answer'] else 'eliminative'
    
    return result

def gen_eval_new(preds):
    # 
    accuracy_total = 0
    count = 0
    
    results = {} 
    
    for pred in preds:
        qid = pred["qid"]
        sent = pred["question"].lower().strip()
        count += 1
        gold = pred["answer"]
        
        prediction = pred["output"]

        accuracy_current = metric_match(exact_match_score, prediction, gold)
        accuracy_total += accuracy_current
        
        dtype = dtype_processing(pred)
        
        if dtype in results:
            results[dtype]['accuracy'] += accuracy_current
            results[dtype]['count'] += 1
        else:
            results[dtype] = {'accuracy': accuracy_current, 'count': 1}
            
    for dtype in results:
        results[dtype]['accuracy'] /= results[dtype]['count']
        
    return results
            

def accuracy(preds, golds, cnn_only):
    count = 0
    correct = 0
    score_exists = False
    if "score" in preds[0].keys():
        score_exists = True
        wrong_score = 0
        correct_score = 0
    for pred, gold in zip(preds, golds):
        if cnn_only and gold["question_source"] != "CNN":
            continue
        prediction = pred["prediction"]
        gold = gold["answer"]
        if prediction == gold:
            correct += 1
            if score_exists:
                correct_score += float(pred["score"])
        else:
            if score_exists:
                wrong_score += float(pred["score"])
        count += 1
    if score_exists:
        return {'accuracy': correct/count, 'correct_score': correct_score/correct, 'wrong_score': wrong_score/(count-correct), 'score': (correct_score + wrong_score)/count}
    
    return {'accuracy': correct/count}

def gen_eval(preds, golds):
    em_total = 0
    f1_total = 0
    count = 0
    
    for pred, gold in zip(preds, golds):
        sent = gold["question_sentence"].lower().strip()
        
        count += 1
        golds = [gold["choices"][int(idx)] for idx in gold["answer"]]
        golds = [' '.join(perm) for perm in list(itertools.permutations(golds))]
        prediction = pred["prediction"]
        em_total += metric_max_over_ground_truths(exact_match_score, prediction, golds)
        f1_total += metric_max_over_ground_truths(f1_score, prediction, golds)
        
    return {'em': em_total/count, 'f1': f1_total/count}
            
def main(
    pred_file: str = '/hdd/hjl8708/workspace/MCQA_option_finetuning/result/Test-AMR_LDA-mixtral_instruct/greedy_preds.json', 
    generate: bool = True, 
):
    with open(pred_file, 'r') as f:
        preds = json.load(f) # {'qid': str, 'question': str, 'output': str, 'answer': str}
        
    if generate:
        results = gen_eval_new(preds)
        
    print(results)


if __name__ == '__main__':
    fire.Fire(main)
