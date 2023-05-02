import numpy as np 
import pandas as pd
import os 
import evaluate
from datasets import load_dataset
import spacy 
import pytextrank
#import gensim

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
    pred = list(example["prediction"].split(" "))
    #print(len(pred))
    ref = list(example["highlights"].split(" "))
    pred = pred[:len(ref)]
    result = metric.compute(predictions=pred, references=ref)
    return result

def main():
    cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = cnn_dailymail['train']
    #metric = evaluate.load('rouge')

    train_dataset_preds = train_dataset.map(eval, num_proc=64)
    results = train_dataset_preds.map(compute_metrics, num_proc=64)
    print(results)


if __name__=='__main__':
    main()