#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import re
import nltk
import numpy as np
from nltk.tag.stanford import StanfordPOSTagger
import os

class MyTokenizer():
    def __init__(self, regex=None):
        self.regex = regex

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            document = str(document)
            document = document.lower()
            for sent in nltk.sent_tokenize(document):
                tokenizer = nltk.RegexpTokenizer(self.regex, flags=re.UNICODE | re.IGNORECASE)
                tokenized_doc += tokenizer.tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

class StanfordPOS():
    def __init__(self, model_filename,jarfile):
        self.model_filename = model_filename
        self.path_to_jar = jarfile
        self.tager=StanfordPOSTagger(model_filename=self.model_filename, path_to_jar=self.path_to_jar)

    def tagger(self, X):
        transformed_X = []
        for doc in X:
            res=self.tager.tag(doc)
            transformed_X.append(np.array(res))
        return transformed_X


if __name__ == '__main__':
    docs = [" Vous allez découvrir la magnifique faune et la flore qui caractérisent la Loire, ses îles ou encore la traversée sous les ponts sur ce parcours",
            " Une ou deux magnifique balade vous attend au bord de l'eau, là où le charme de la Loire et vous surprend"]


    reg_words = r'''(?x)
            aujourd'hui # exception 1
            |\w+-\w+    # les mots composes
            | \w'       # les contractions d', l', j', t', s'
            |\w+        # les mots simples
            '''
    tokenizer=MyTokenizer(reg_words)
    doc_tokens=tokenizer.fit_transform(docs)

    root = os.path.dirname(__file__)
    project_root = root + os.sep + '..' + os.sep + '..'

    # Stanford POSTagger
    stanford_dir = project_root + os.sep + 'data' + os.sep + 'stanford-postagger-full-2017-06-09'
    modelfile = stanford_dir + os.sep + 'models' + os.sep + 'french.tagger'
    jarfile = stanford_dir + os.sep + 'stanford-postagger.jar'

    POS_tokens=StanfordPOS(model_filename=modelfile,jarfile=jarfile)
    print(POS_tokens.tagger(doc_tokens))

