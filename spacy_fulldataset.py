import numpy as np 
import pandas as pd
import os 
import evaluate
from datasets import load_dataset
import spacy 
import pytextrank
#import gensim
from rouge import Rouge

#spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("textrank")
metric = evaluate.load('rouge')

def eval(example):
    """
        https://derwen.ai/docs/ptr/sample/
    """ 
    doc = nlp(example["article"])
    tr = doc._.textrank
    summary = ""
    for sent in tr.summary(limit_phrases=10, limit_sentences=1):
        summary += str(sent)
    example["prediction"] = summary
    return example

def compute_metrics(example):
    """
    """
    pred = example["prediction"]
    ref = example["highlights"]
    #pred = pred[:len(ref)]
    result = metric.compute(predictions=pred, references=ref)
    return result

def _compute_metrics(example):
    rouge = Rouge()
    pred = example["prediction"]
    ref = example["highlights"]

    score = rouge.get_scores(pred, ref)
    return score[0]
def main():
    cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = cnn_dailymail['train']
    #metric = evaluate.load('rouge')

    train_dataset_preds = train_dataset.map(eval, num_proc=64)
    results = train_dataset_preds.map(_compute_metrics, num_proc=64)
    rouge_results = {}

    print(results["rouge-1"])


if __name__=='__main__':
    main()