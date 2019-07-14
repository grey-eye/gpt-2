import os
import pandas as pd
import numpy as np

def get_dataset():
    dataset_path = 'datasetSentences.txt'
    label_path = 'sentiment_labels.txt'
    split_path = 'datasetSplit.txt'
    dictionary_path = 'dictionary.txt'
    sentences = pd.read_csv(dataset_path, sep='\t')
    labels = pd.read_csv(label_path, sep='|')
    splits = pd.read_csv(split_path, sep=',')
    dictionary = {}
    with open(dictionary_path) as fin:
        for line in fin.readlines():
            tokens = line.strip().split('|')
            dictionary[tokens[0]] = tokens[1]
    sentences['phrase ids'] = sentences.apply(lambda x: int(dictionary.get(x['sentence'], 0)), axis=1)
    dataset = sentences.merge(labels).merge(splits)
    return dataset

def remove_newlines(sentence):
    if sentence is None:
        return sentence
    else:
        return str(sentence).replace('\n', ' ').replace('\r', '')

dataset = get_dataset()
dataset['sentence'] = dataset['sentence'].apply(remove_newlines)
train = dataset[dataset['splitset_label']==1][['sentiment values', 'sentence']]
test = dataset[dataset['splitset_label']==2][['sentiment values', 'sentence']]
dev = dataset[dataset['splitset_label']==3][['sentiment values', 'sentence']]

train.to_csv('data/train.tsv', sep='\t', index=False)
test.to_csv('data/test.tsv', sep='\t', index=False)
dev.to_csv('data/dev.tsv', sep='\t', index=False)

