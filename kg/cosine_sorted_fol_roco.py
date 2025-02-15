import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

# Rule application and processing functions remain unchanged
def remove_duplicate(kg_triplets):
    res = []
    [res.append(x) for x in kg_triplets if x not in res]
    return res

def parse_triple(kg_triplets):
    kg_len = len(kg_triplets)
    empty = ["_NAF_H", "_NAF_R", "_NAF_O"]
    if kg_len <= 20:
        tt = 20 - kg_len
        for _ in range(tt):
            kg_triplets.append(empty)
    return kg_triplets

def apply_rules_to_kg(kg_triplets):
    co_occurs_triplets, prevent_triplets = [], []
    treatment_triplets, diagnosis_triplets = [], []
    conjunction_triplets, disjunction_triplets = [], []

    for triplet in kg_triplets:
        if triplet[1] == "co-occurs_with":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[2] and other_triplet[1] == "affects":
                    co_occurs_triplets.append([triplet[0], "affects", other_triplet[2]])

        if triplet[1] == "prevents":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[2] and other_triplet[1] == "causes":
                    prevent_triplets.append([triplet[0], "prevents", other_triplet[2]])

        if triplet[1] == "treats":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[2] and other_triplet[1] == "isa":
                    treatment_triplets.append([triplet[0], "treats", other_triplet[2]])

        if triplet[1] == "diagnoses":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[0] and other_triplet[1] == "interacts_with":
                    diagnosis_triplets.append([other_triplet[2], "diagnoses", triplet[0]])

        if triplet[1] == "co-occurs_with":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[0] and other_triplet[1] == "affects":
                    conjunction_triplets.append([triplet[2], "co-occurs_with", other_triplet[2]])

        if triplet[1] == "prevents":
            X, Y = triplet[0], triplet[2]
            for other_triplet in kg_triplets:
                if other_triplet[1] == "causes" and other_triplet[0] == Y:
                    Z = other_triplet[2]
                    disjunction_triplets.append([X, "prevents", Z])
                    disjunction_triplets.append([X, "causes", Z])

    return (
        remove_duplicate(parse_triple(remove_duplicate(co_occurs_triplets))),
        remove_duplicate(parse_triple(remove_duplicate(prevent_triplets))),
        remove_duplicate(parse_triple(remove_duplicate(treatment_triplets))),
        remove_duplicate(parse_triple(remove_duplicate(diagnosis_triplets))),
        remove_duplicate(parse_triple(remove_duplicate(conjunction_triplets))),
        remove_duplicate(parse_triple(remove_duplicate(disjunction_triplets))),
    )

def get_bert_based_similarity(sentence_pairs, model, tokenizer):
    similarities = {}
    inputs_1 = tokenizer(sentence_pairs[0], return_tensors="pt")
    sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)

    for count, triple in enumerate(sentence_pairs[1]):
        inputs_2 = tokenizer(" ".join(triple), return_tensors="pt")
        sent_2_embed = np.mean(model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)
        similarities[" ".join(triple)] = dot(sent_1_embed, sent_2_embed) / (norm(sent_1_embed) * norm(sent_2_embed))
    return similarities

def sorted_triple(combined_question, rule_kg_triples, model, tokenizer):
    sentence_pairs = [combined_question, rule_kg_triples]
    sim_final_dict = get_bert_based_similarity(sentence_pairs, model, tokenizer)

    triple_n_sim_final_dic = {k: t for k, t in zip(sim_final_dict.keys(), rule_kg_triples)}
    sorted_dict = {k: v for k, v in sorted(sim_final_dict.items(), key=lambda item: item[1], reverse=True)}

    return [triple_n_sim_final_dic[k] for k in sorted_dict.keys()]


# Load the dataset

input_file ="../ROCO/roco data/test_with_kg.json"
output_file = "../ROCO/roco data/test_with_rule_kg.json"



with open(input_file, "r") as f:
    data = json.load(f)

print("Data loaded.")

# Load PubMed BERT model
from transformers import AutoModel, AutoTokenizer

pubmed_bert_model = AutoModel.from_pretrained(
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    trust_remote_code=True
)
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained(
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    trust_remote_code=True
)

processed_data = []

for num, item in enumerate(data):
    print(f"Processing item {num + 1}/{len(data)}")
    caption = item["caption"]
    kg_triplets = item["kg_triples"]

    # Apply rules to categorize triples
    r1, r2, r3, r4, r5, r6 = apply_rules_to_kg(kg_triplets)
    print(len(r1), len(r2), len(r3), len(r4), len(r5), len(r6))

    r1_sorted = sorted_triple(caption, r1, pubmed_bert_model, pubmed_bert_tokenizer)
    r2_sorted = sorted_triple(caption, r2, pubmed_bert_model, pubmed_bert_tokenizer)
    r3_sorted = sorted_triple(caption, r3, pubmed_bert_model, pubmed_bert_tokenizer)
    r4_sorted = sorted_triple(caption, r4, pubmed_bert_model, pubmed_bert_tokenizer)
    r5_sorted = sorted_triple(caption, r5, pubmed_bert_model, pubmed_bert_tokenizer)
    r6_sorted = sorted_triple(caption, r6, pubmed_bert_model, pubmed_bert_tokenizer)

    processed_sample = {
        "id": item["id"],
        "file_name": item["file_name"],
        "caption": item["caption"],
        "keywords": item["keywords"],
        "cuis": item["cuis"],
        "semtypes": item["semtypes"],
        "kg_triples": kg_triplets,
        "rule_1": r1_sorted,
        "rule_2": r2_sorted,
        "rule_3": r3_sorted,
        "rule_4": r4_sorted,
        "rule_5": r5_sorted,
        "rule_6": r6_sorted,
    }

    processed_data.append(processed_sample)


# Save the processed data
with open(output_file, "w") as f:
    json.dump(processed_data, f, indent=4)

print("Processed data saved.")
