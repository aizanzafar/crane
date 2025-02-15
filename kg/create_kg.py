# #https://github.com/DATEXIS/UMLSParser follow this for creating UMLS-FILE

from quickumls import QuickUMLS
import re
import pandas as pd

import json
import string

import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 200)



quickumls_fp= "../UMLS/umls-extract/umls-final"
relationpath='../UMLS/umls-extract/NET/SRSTRE1'
defenitionpath='../UMLS/umls-extract/NET/SRDEF'


matcher = QuickUMLS(quickumls_fp, threshold=0.98, window=3,min_match_length=1)

final_relation=[]
result_list=[]
global_terms=[]

contexts=[]

def output_of_umls(text):
    terms=[]
    results = matcher.match(text, best_match=True, ignore_syntax=False)
    for result in results:
      for x in result:
        terms.append({'term': x['term'].lower(), 'cui': x['cui'], 'semtype':x['semtypes']})
    return terms

def readSemanticDefination(defenitionpath, sem_type):
    with open(defenitionpath,'r') as f:
        for line in f:
            typ=line.split('|')
            if typ[1] == sem_type:
                return typ[2]

def readSemanticRelations(relationpath, sem_type, first,final_concept):
    final_relation=[]
    with open(relationpath,'r') as f:
        for line in f:
            type=line.split('|')
            if type[0]==sem_type:
                #print("yessss")
                #if str(type[1]) in relationship:
                    #print("yes")
                pp=SemType_to_CUI(type[2],final_concept)
                if len(pp)!= 0:
                    for x in pp:
                        if x[1] != first:
                            final_relation.append((readSemanticDefination(defenitionpath,type[1]),x[1]))
    return final_relation

def SemType_to_CUI(sem_type,final_concept):
    triplets=[]
    for x in final_concept:
        for i in x['semtype']:
            if i==sem_type:
                triplets.append((x['cui'],x['term']))
    return list(set([i for i in triplets]))

def remove_duplicate_triplets(final_relation):
    return list(set([i for i in final_relation]))


def remove_duplicate_term(terms):
    res_list=[]
    for umls_term in terms:
        if len(res_list)==0:
            res_list.append(umls_term)
        if umls_term not in res_list:
            res_list.append(umls_term)
    return res_list


def terms_output(text):
    terms= output_of_umls(text)
    # remove duplicate terms 
    terms = remove_duplicate_term(terms)
    return terms


