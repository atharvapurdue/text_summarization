import pandas as pd
from rouge import Rouge
import numpy as np

def compute_f1_score(series: pd.Series, nlp, num=20):
    """Computes an F1 score from a given dataset, 
    up to the number of records specified. 

    Arguments
    ---------
    series : pandas Series
    Series containing the 
    """
    rouge = Rouge()
    scores = []
    ans = ""

    for i in range(20):
        doc = nlp(series[i])
        for j in doc._.textrank.summary(limit_phrases=10, limit_sentences=1):
            ans+=str(j)
            score = rouge.get_scores(ans, series[i])
            score = score[0]
            scores.append(score["rouge-l"]["f"])

    return np.mean(scores)