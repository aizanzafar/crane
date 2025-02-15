import json
import os
from io import open

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english')

from load_covidqa_dic import*			# return_kg and n_hops for 2
from create_kg import*					# umls kg creation
from biobert_embedding_cosine_sim import* #cosine similarity between question n triplets
# from preprocess import*
from preprocess_2hops import*

pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')

input_file= "../ROCO/roco data/test.json"
output_file ="../ROCO/roco data/test_with_kg.json"

with open(input_file, "r") as f:
    roco_data = json.load(f)

# Transform kg_triple into kg_triples
updated_data = []
for i, item in enumerate(roco_data):
    print(f"Processing item {i + 1}/{len(roco_data)}...")
    caption = item["caption"]
    
    # Generate knowledge graph triples using the caption
    kg_triple_dict = context_kg_triple(caption)
    
    # Convert kg_triple to kg_triples format
    kg_triples_list = []
    for head, relations in kg_triple_dict.items():
        for relation, tail in relations:
            kg_triples_list.append([head, relation, tail])
    
    # Add the transformed KG triples to the sample
    item["kg_triples"] = kg_triples_list
    updated_data.append(item)

# Save the updated dataset with transformed KG triples
with open(output_file, "w") as f:
    json.dump(updated_data, f, indent=4)

print(f"Updated dataset saved to {output_file}")

