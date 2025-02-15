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

from load_covidqa_dic import*			# return_kg and n_hops for 2
from create_kg import*					# umls kg creation
from biobert_embedding_cosine_sim import* #cosine similarity between question n triplets
# from preprocess import*
from preprocess_2hops import*

import json


def process_context_kg(context_kg):
    """
    Transform the context_kg dictionary into a triplet list format.
    """
    triplets = []
    for key, relations in context_kg.items():
        for relation, target in relations:
            triplets.append([key, relation, target])
    return triplets

def process_data(input_file, output_file):
    """
    Process the input JSON file to extract context KG triplets and save them to the output file.
    """
    # Load the merged JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    processed_data = []
    start = 0

    for item in data:
        print(f"############################  {start}  ############################")
        url = item.get("url")
        caption = item.get("caption", "Caption not found")
        finding = item.get("finding", "Finding not found")
        other_info = item.get("other_info", "")

        # Merge caption and other_info for context
        context = f"{caption} {other_info}".strip()

        # Skip sample if both caption and finding are "not found"
        if caption == "Caption not found" and finding == "Finding not found":
            continue

        # Create context_kg
        context_kg = context_kg_triple(context)
        # print(f"context_kg size: {len(context_kg)}")
        # print(context_kg)

        # Convert context_kg dictionary to triplet list format
        triplets = process_context_kg(context_kg)

        print(triplets)
        print(f"Triplets size: {len(triplets)}")
        
        # Add processed sample with kg_triplets
        processed_sample = {
            "url": url,
            "caption": caption,
            "finding": finding,
            "other_info": other_info,
            "image_name": item.get("image_name"),
            "kg_triplets": triplets,
        }

        start += 1
        processed_data.append(processed_sample)

    # Save processed data to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    print(f"Processed data saved to {output_file}. Total samples: {len(processed_data)}")


# Input and output file paths
input_file = "vqa_rad/vqa-rad.json"  # Input file
output_file = "vqa_rad/processed_vqa_with_kg.json"  # Output file with kg_triplets

# Process the data
process_data(input_file, output_file)
