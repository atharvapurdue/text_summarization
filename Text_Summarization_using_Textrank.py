import numpy as np
import pandas as pd
import os
from summa.summarizer import summarize
from summa import keywords
from pprint import PrettyPrinter #print in a pretty way 
pp = PrettyPrinter()


def summary_for_article(num, df, prin=False):
    
    article = df.article[num]
    
    # get summary using TextRank
    summary = summarize(article, ratio=0.1)
    
    # get important phrases using TextRank
    phrases = keywords.keywords(article, ratio=0.1).split('\n')
    phrases_and_ranks = [(phrase, None) for phrase in phrases]
    
    if prin:
        print(article)
        print("\n_______ to ______\n")
        print(summary)
        print("\n_______ important phrases ______\n")
       
    pp.pprint(phrases_and_ranks[:10])
    return summary


def load_datasets(train_path, test_path, validation_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_validation = pd.read_csv(validation_path)

    # concat all of the dfs together
    df = pd.concat([df_train, df_test, df_validation], ignore_index=True)

    del df_train, df_test, df_validation

    return df


def main():
    df = load_datasets('C:\\Users\\navya\\Downloads\\train.csv', 'C:\\Users\\navya\\Downloads\\test.csv', 'C:\\Users\\navya\\Downloads\\validation.csv')
    for i in range(0,21):
        print("\n....",i,"")
        summary_for_article(i, df, True)


if __name__=='__main__':
    main()
