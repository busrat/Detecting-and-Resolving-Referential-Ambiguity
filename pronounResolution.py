# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 00:05:30 2021

@author: busra
"""

from pycorenlp import StanfordCoreNLP
import pandas as pd
nlp = StanfordCoreNLP('http://localhost:9000')

df = pd.read_csv('training_set2.csv', sep=r'\s*,\s*', error_bad_lines=False)
data = [item.strip('"') for item in df['sent']]

def resolve(sent):
    for coref in sent['corefs']:
        mentions = sent['corefs'][coref]
        antecedent = mentions[0]
        for mention in mentions[1:]:
            if mention['type'] == 'PRONOMINAL':
                sentence = mention['sentNum']
                token = mention['startIndex'] - 1
                sent['sentences'][sentence - 1]['tokens'][token]['word'] = antecedent['text']

def disambiguate(sent):
    nonambiguousWord = ""
    possessives = ['mine', 'yours', 'hers', 'his', 'ours', 'yours', 'theirs', 'its']
    for sentence in sent['sentences']:
        for token in sentence['tokens']:
            word = token['word']
            if token['lemma'].lower() in possessives or token['pos'] == 'PRP$':
                word += "'s"
            word += token['after']
            nonambiguousWord += word
    return nonambiguousWord

disambiguation = pd.DataFrame(columns = ['Before', 'After'])

for d, i in zip(data,range(len(data))):
    annotatiton = nlp.annotate(d, properties= {'annotators':'dcoref','outputFormat':'json','ner.useSUTime':'false'})
    resolve(annotatiton) 
    disambiguation.loc[i] = d, disambiguate(annotatiton)
