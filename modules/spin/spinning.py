#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import os
from modules.spin import linguistic_analyze as la
from modules.utils import random_data as rd

class CreateText():
    def __init__(self, lexique, dico_morpho):
        self.lexique = lexique
        self.dico_morpho= dico_morpho

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        token_prec = ''
        liste_doc_variantes=[]
        for i in range(0,len(X['tokens'])):
            doc=X['tokens'][i]
            liste_variantes = []
            liste_variantes.append(np.array(str(X['original'][i])))
            for token in doc :

                # les ADJ
                if token[1]=='ADJ':
                    if self.lexique['LEMME'].isin([token[0]]).any():
                        use = self.lexique['USE'][self.lexique['LEMME'] == token[0]].values[0]
                        pol = self.lexique['POLARITE'][self.lexique['LEMME'] == token[0]].values[0]
                        test = self.lexique['LEMME'][(self.lexique['USE'] == use) & (self.lexique['POLARITE'] == pol)]
                        liste_syn = test.tolist()
                        gender = self.dico_morpho['gender'][self.dico_morpho['forme'] == token_prec].values[0]
                        number = self.dico_morpho['number'][self.dico_morpho['forme'] == token_prec].values[0]
                        for syn in liste_syn:
                            syn_accorde = self.dico_morpho['forme'][
                                (self.dico_morpho['gender'] == gender) & (self.dico_morpho['lemme'] == syn) & (
                                        self.dico_morpho['number'] == number)].values
                            if syn_accorde != '':
                                phrase = re.sub(token[0], syn_accorde[0], str(X['original'][i]))
                                STARTVOYELLE = self.lexique['STARTVOYELLE'][self.lexique['LEMME'] == syn].values[0]
                                POSITIONADJ = self.lexique['POSITION'][self.lexique['LEMME'] == syn].values[0]

                                if STARTVOYELLE:
                                    if token_prec == 'le' or token_prec == 'la':
                                        phrase = re.sub(token_prec + ' ' + syn_accorde[0], "l'"+ syn_accorde[0], phrase)
                                if POSITIONADJ == 'AP':

                                    if token_prec == 'le' or token_prec == 'la' or token_prec =="un" or token_prec =="une":
                                        phrase = re.sub(syn_accorde[0] + ' (\w+?)[ !]', "\\1 " + syn_accorde[0], phrase)
                                if POSITIONADJ == 'AV':

                                    if token_prec != 'le' or token_prec != 'la' or token_prec !="un" or token_prec !="une":
                                        phrase = re.sub(' (\w+?) ' + syn_accorde[0], ' ' + syn_accorde[0] + " \\1", phrase)
                                # print(phrase)
                                liste_variantes.append((np.array(phrase)))

               # les VER
                if token[1] == 'V' or token[1] =='VINF':
                    if self.lexique['LEMME'].isin([token[0]]).any():
                        use = self.lexique['USE'][self.lexique['LEMME'] == token[0]].values[0]
                        pol = self.lexique['POLARITE'][self.lexique['LEMME'] == token[0]].values[0]
                        test = self.lexique['LEMME'][
                            (self.lexique['USE'] == use) & (self.lexique['POLARITE'] == pol)]
                        liste_syn = test.tolist()
                        tense = self.dico_morpho['tense'][self.dico_morpho['forme'] == token[0]].values[0]
                        for syn in liste_syn:

                            syn_accorde = self.dico_morpho['forme'][
                                (self.dico_morpho['tense'] == tense)
                                & (self.dico_morpho['lemme'] == syn)
                                & (syn != token[0])].values

                            if syn_accorde != '':
                                phrase = re.sub(token[0], syn_accorde[0], str(X['original'][i]))
                                liste_variantes.append((np.array(phrase)))
                                # print(liste_variantes)

                # les NOM
                if token[1] == 'NC':
                    if self.lexique['LEMME'].isin([token[0]]).any():
                        use = self.lexique['USE'][self.lexique['LEMME'] == token[0]].values[0]
                        pol = self.lexique['POLARITE'][self.lexique['LEMME'] == token[0]].values[0]
                        test = self.lexique['LEMME'][
                            (self.lexique['USE'] == use) & (self.lexique['POLARITE'] == pol)]
                        liste_syn = test.tolist()
                        number = self.dico_morpho['number'][self.dico_morpho['forme'] == token_prec].values[0]
                        for syn in liste_syn:
                            syn_accorde = self.dico_morpho['forme'][
                                 (self.dico_morpho['lemme'] == syn) & (
                                        self.dico_morpho['number'] == number) & (
                                        syn != token[0])].values
                            if syn_accorde != '':
                                phrase = re.sub(token[0], syn_accorde[0], str(X['original'][i]))
                                STARTVOYELLE = self.lexique['STARTVOYELLE'][self.lexique['LEMME'] == syn].values[0]

                                if STARTVOYELLE:
                                    if token_prec == 'le' or token_prec == 'la':
                                        phrase = re.sub(token_prec + ' ' + syn_accorde[0], "l'" + syn_accorde[0], phrase)
                                        print(phrase)
                                liste_variantes.append((np.array(phrase)))
                token_prec = token[0]
            liste_doc_variantes.append(np.array(liste_variantes))

        return np.array(liste_doc_variantes)

    def transform_ville(self, X):
        liste_villes=['Lyon','Fontainebleau','Orléans','Paris','Rennes','Nancy','Bordeaux','Nice','Cannes','Valence','Lille','Montpellier']
        for i in range(0,len(X)):
            X[i]=re.sub('\[Ville\]',liste_villes[i], str(X[i]))
            X[i]=re.sub("de ([AEIOUY])","d'\\1", str(X[i]))
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def construct_text(self, dico, num):
        df=pd.DataFrame(columns=['textes'])
        for i in range (0,num):
            texte=dico['Debut'][i] + ' ' + dico['AvantFin'][i] + ' ' + dico['Fin'][i]
            df.loc[i]=texte
        return df

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    project_root = root + os.sep + '..' + os.sep + '..'
    pd.options.display.max_colwidth = 400
    num=9

    # Lexiques
    lexique = pd.read_csv(project_root + '/data/lexique.txt', sep='\t', encoding='utf8')
    lexique_ira = pd.read_csv(project_root + '/data/lexique_fr.txt', sep='\t', encoding='utf8')

    # Stanford POSTagger
    stanford_dir = project_root + os.sep + 'data' + os.sep + 'stanford-postagger-full-2017-06-09'
    modelfile = stanford_dir + os.sep + 'models' + os.sep + 'french.tagger'
    jarfile = stanford_dir + os.sep + 'stanford-postagger.jar'


    # Lecture inputs
    base_textes=pd.read_csv(project_root + os.sep + 'data' + os.sep + 'base_textes.txt', sep='\t', encoding='utf8')
    print(base_textes.head(5))
    liste_villes=pd.read_csv(project_root + os.sep + 'data' + os.sep + 'villes.txt', sep='\t', encoding='utf8')

    reg_words = r'''(?x)
              aujourd'hui # exception 1
              |\w+-\w+    # les mots composes
              | \w'       # les contractions d', l', j', t', s'
              |\w+        # les mots simples
              '''

    # --------- selection beforeEOA ---------
    dataBefEOA=base_textes['BASE'][base_textes['LOC']=='BefEOA']
    docs=rd.select_random_data(dataBefEOA,6)

    # tokenization
    tokenizer = la.MyTokenizer(reg_words)
    doc_tokens = tokenizer.fit_transform(docs)

    # exctrat POS
    POS_tokens = la.StanfordPOS(model_filename=modelfile, jarfile=jarfile)
    tokens_POS = POS_tokens.tagger(doc_tokens)

    # docs originaux
    df=pd.DataFrame(docs, columns=['original'])
    df['tokens']=tokens_POS

    # spin
    resu=CreateText(lexique=lexique, dico_morpho=lexique_ira)
    beforeendoftext=resu.transform(df)
    # unlist
    flatten = [val for sublist in beforeendoftext for val in sublist]
    df_beforeendoftext=pd.DataFrame(flatten, columns=['beforeendoftext'])

    # --------- selection BOA ---------

    dataBOA = base_textes['BASE'][base_textes['LOC'] == 'BOA']
    docs = rd.select_random_data(dataBOA, num)

    # tokenization
    tokenizer = la.MyTokenizer(reg_words)
    doc_tokens = tokenizer.fit_transform(docs)

    # exctrat POS
    POS_tokens = la.StanfordPOS(model_filename=modelfile, jarfile=jarfile)
    tokens_POS = POS_tokens.tagger(doc_tokens)

    # docs originaux
    df = pd.DataFrame(docs, columns=['original'])
    df['tokens'] = tokens_POS

    # spin
    resu = CreateText(lexique=lexique, dico_morpho=lexique_ira)
    beginoftext = resu.transform(df)
    df_beginoftext=pd.DataFrame(beginoftext, columns=['beginoftext'])

    # --------- selection EOA ---------
    dataBefEOA=base_textes['BASE'][base_textes['LOC']=='EOA']
    docs=rd.select_random_data(dataBefEOA,6)

    # tokenization
    tokenizer = la.MyTokenizer(reg_words)
    doc_tokens = tokenizer.fit_transform(docs)

    # exctrat POS
    POS_tokens = la.StanfordPOS(model_filename=modelfile, jarfile=jarfile)
    tokens_POS = POS_tokens.tagger(doc_tokens)

    # docs originaux
    df=pd.DataFrame(docs, columns=['original'])
    df['tokens']=tokens_POS

    # spin
    resu=CreateText(lexique=lexique, dico_morpho=lexique_ira)
    endoftest=resu.transform(df)
    # unlist
    flatten = [val for sublist in endoftest for val in sublist]
    df_endoftext=pd.DataFrame(flatten, columns=['endoftext'])

    # melange des phrases
    liste_radom_df_beforeendoftext = rd.select_random_data(df_beforeendoftext.beforeendoftext, len(df_beforeendoftext.beforeendoftext))
    df_beforeendoftext = pd.DataFrame(liste_radom_df_beforeendoftext, columns=['beforeendoftext'])

    # melange des phrases
    liste_radom_df_endoftext = rd.select_random_data(df_endoftext.endoftext, len(df_endoftext.endoftext))
    df_endoftext = pd.DataFrame(liste_radom_df_endoftext, columns=['endoftext'])

    # Creation dict avec differentes phrases
    mondict = {'Debut': df_beginoftext.beginoftext, 'AvantFin': df_beforeendoftext.beforeendoftext, 'Fin': df_endoftext.endoftext}
    concat=resu.construct_text(mondict, num)
    # remplacer ville par nom de villes
    FIN = resu.transform_ville(concat.textes)
    print(FIN)

