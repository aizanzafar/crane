import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english')

from load_covidqa_dic import*
from create_kg import*
from biobert_embedding_cosine_sim import*



def context_kg_triple(context):
	final_concept= terms_output(context)
	final_dict=dict()
	count=0
	for x in final_concept:
	    c_name=x['term']
	    for i in x['semtype']:
	        # print(count)
	        count=count+1
	        if c_name in final_dict.keys():
	            triplet = remove_duplicate_triplets(readSemanticRelations(relationpath,i,x['term'],final_concept))
	            final_dict[c_name].extend(triplet)
	            final_dict[c_name] = list(set(final_dict[c_name]))
	        else:
	            triplet = remove_duplicate_triplets(readSemanticRelations(relationpath,i,x['term'],final_concept))
	            final_dict[c_name] = triplet
	return final_dict

def sorted_triple(question,context_kg,pubmed_bert_model,pubmed_bert_tokenizer):
	q= re.sub(r'[^\w\s]','',question)
	q_concept = terms_output(q)
	q_tokens=[]
	for x in q_concept:
		q_tokens.append(x['term'])
	q_tokens= list(set(q_tokens))
	final_kg_triplet = n_hops(q_tokens,context_kg,n=2)

	sentence_pairs = [question,final_kg_triplet]
	sim_final_dict= get_bert_based_similarity(sentence_pairs, pubmed_bert_model, pubmed_bert_tokenizer)

	triple_n_sim_final_dic=dict()
	for k,t in zip(sim_final_dict.keys(),final_kg_triplet):
		triple_n_sim_final_dic[k]=t

	sorted_dict = {k: v for k, v in sorted(sim_final_dict.items(), key=lambda item: item[1], reverse=True)}

	sorted_triple_list=[]
	for k,v in sorted_dict.items():
		sorted_triple_list.append(triple_n_sim_final_dic[k])

	return sorted_triple_list


