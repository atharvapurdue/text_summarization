import numpy as np 
import pandas as pd
import os 

import spacy 
import pytextrank

from pprint import PrettyPrinter #print in a pretty way 
pp = PrettyPrinter()

def summary_for_article(num,df,nlp,prin=False):
    
    ans = "" # collecting the summary from the generator
    doc = nlp(df.article[num]) #apply the pipeline
    
    for i in doc._.textrank.summary(limit_phrases=10, limit_sentences=1): #get the summary
        ans+=str(i)
        
    phrases_and_ranks = [ (phrase.chunks[0], phrase.rank) for phrase in doc._.phrases] # get important phrases
    
    if prin: # print
        print(df.article[num])
        print("\n_______ to ______\n")
        print(ans)
        print("\n_______ important phrases ______\n")
        pp.pprint(phrases_and_ranks[:10])
        
    return ans


def load_datasets(path_to_data_folder):
    df_train = pd.read_csv(path_to_data_folder+'train.csv')
    df_test = pd.read_csv(path_to_data_folder+'test.csv')
    df_validation = pd.read_csv(path_to_data_folder+'validation.csv')

    # concat all of the dfs together
    df = pd.concat([df_train, df_test, df_validation], ignore_index=True)

    del df_train, df_test, df_validation

    return df

def main():
    df = load_datasets('data/')

    # make sure to run this command first
    #spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textrank")
    for i in range(0,21):
        print("\n....",i,"")
        summary_for_article(i, df, nlp, True)


if __name__=='__main__':
    main()