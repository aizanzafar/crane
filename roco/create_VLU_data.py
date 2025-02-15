import json

# Input and output file paths
input_file = "vqa_rad/final_vqa_rad.json"
output_file = "vqa_rad/final_vqa_rad_no_rules.json"

# Load the dataset
with open(input_file, "r") as f:
    data = json.load(f)

# Process the dataset to remove the rules
processed_data = []
for entry in data:
    # Create a new dictionary with only the desired fields
    new_entry = {
        "qid": entry["qid"],
        "phrase_type": entry["phrase_type"],
        "qid_linked_id": entry["qid_linked_id"],
        "image_case_url": entry["image_case_url"],
        "image_name": entry["image_name"],
        "image_organ": entry["image_organ"],
        "evaluation": entry["evaluation"],
        "question": entry["question"],
        "question_relation": entry["question_relation"],
        "question_frame": entry["question_frame"],
        "question_type": entry["question_type"],
        "answer": entry["answer"],
        "answer_type": entry["answer_type"],
        "kg_triples": entry["kg_triples"]
    }
    
    # Add question_rephrase if it exists
    if "question_rephrase" in entry:
        new_entry["question_rephrase"] = entry["question_rephrase"]
    else:
        new_entry["question_rephrase"] = "NULL"
    
    processed_data.append(new_entry)

print(len(processed_data))
# Save the processed data
with open(output_file, "w") as f:
    json.dump(processed_data, f, indent=4)

print(f"Processed dataset saved to {output_file}")
