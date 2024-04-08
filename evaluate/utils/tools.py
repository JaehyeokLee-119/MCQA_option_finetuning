import json, datetime, string
from collections import Counter
        
def add_today(sentence, date):
    date = datetime.datetime.strptime(date, '%Y/%m/%d')
    date = date.strftime("%B %d, %Y")
    sentence = "Today is {}. ".format(date) + sentence
    return sentence

def normalize_answer(s):
    # should we keep those counter removal? 
    def white_space_fix(text):
        return ' '.join(text.split()) # 두 개 이상 나오는 white space를 하나로 만들어줌

    def remove_punc(text):
        exclude = set(string.punctuation) # string.punctuation 제거 : !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ 
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric, prediction, ground_truths):
    # ground_truths에 속하는 모든 경우의 수들 중에 prediction이 포함되는가? -> 이렇게 구현하면 안될듯
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def metric_match(metric, prediction, ground_truth):
    score = metric(prediction, ground_truth)
    return score
