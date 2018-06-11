#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import nltk
import pandas as pd
import re
from nltk.tag.stanford import StanfordPOSTagger

from modules.spin import linguistic_analyze as la

# -- CONFIG --
# logic=os.path.dirname(sys.modules['__main__'].__file__)
root=os.path.dirname(__file__)
project_root=root + os.sep + '..'

sys.path.append(os.environ['JAVAHOME'])

# -- RESSOURCES --

# Stanford POSTagger
stanford_dir = project_root + os.sep + 'data' + os.sep + 'stanford-postagger-full-2017-06-09'
modelfile = stanford_dir + os.sep + 'models' + os.sep + 'french.tagger'
jarfile = stanford_dir + os.sep + 'stanford-postagger.jar'

# Lexiques
lexique = pd.read_csv(project_root + '/data/lexique.txt', sep='\t', encoding='utf8')
lexique_ira = pd.read_csv(project_root + '/data/lexique_fr.txt', sep='\t', encoding='utf8')



if __name__ == '__main__':

    docs = [" Vous allez découvrir la magnifique faune et la flore qui caractérisent la Loire, ses îles ou encore la traversée sous les ponts sur ce parcours",
            " Une ou deux magnifique balade vous attend au bord de l'eau, là où le charme de la Loire et vous surprend"]

    reg_words = r'''(?x)
            aujourd'hui # exception 1
            |\w+-\w+    # les mots composes
            | \w'       # les contractions d', l', j', t', s'
            |\w+        # les mots simples
            '''

    tokenizer=la.MyTokenizer(reg_words)
    doc_tokens=tokenizer.fit_transform(docs)
    print(doc_tokens)

    POS_tokens=la.StanfordPOS(model_filename=modelfile,jarfile=jarfile)
    tokens_POS=POS_tokens.tagger(doc_tokens)
    print(tokens_POS)


    print(lexique_ira.head(10))
    token_prec=''
    for token in tokens_POS :
        if lexique['LEMME'].isin([token[0]]).any():
            use=lexique['USE'][lexique['LEMME']==token[0]].values[0]
            pol = lexique['POLARITE'][lexique['LEMME'] == token[0]].values[0]
            print(pol)
            test=lexique['LEMME'][(lexique['USE']==use) & (lexique['POLARITE']==pol)]
            liste_syn=test.tolist()
            gender = lexique_ira['gender'][lexique_ira['forme'] == token_prec].values[0]
            number = lexique_ira['number'][lexique_ira['forme'] == token_prec].values[0]
            print(gender)
            for syn in liste_syn :
                syn_accorde = lexique_ira['forme'][(lexique_ira['gender']==gender) & (lexique_ira['lemme']==syn) & (lexique_ira['number']==number)].values
                if syn_accorde!='':
                    phrase = re.sub(token[0], syn_accorde[0], doc)
                    print(phrase)


            print(token_prec)
        token_prec = token[0]



