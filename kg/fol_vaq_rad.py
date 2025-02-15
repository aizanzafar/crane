import json

"""
# 1. rule of co-coocurance
co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
#  2. Rule of Prevention and Causation:
prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
# 3. Rule of Treatment and Classification:
treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
# 4. Rule of Diagnosis and Interaction:
diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
# 5. Rule of Conjunction:
co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
# 6. Rule of Disjunction:
(prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))

"""

def remove_duplicate(kg_triplets):
    """
    Remove duplicate triplets.
    """
    res = []
    [res.append(x) for x in kg_triplets if x not in res]
    return res


def parse_triple(kg_triplets):
    """
    Parse triplets to ensure each has at least 20 entries, filling with placeholders if needed.
    """
    kg_len = len(kg_triplets)
    empty = ["_NAF_H", "_NAF_R", "_NAF_O"]
    if kg_len <= 20:
        tt = 20 - kg_len
        for _ in range(tt):
            kg_triplets.append(empty)
    return kg_triplets


def apply_rules_to_kg(kg_triplets):
    """
    Apply logical rules to the knowledge graph triplets and return the parsed results.
    """
    co_occurs_triplets = []
    prevent_triplets = []
    treatment_triplets = []
    diagnosis_triplets = []
    conjunction_triplets = []
    disjunction_triplets = []

    # 1. Rule of Co-occurrence
    for triplet in kg_triplets:
        if triplet[1] == "co-occurs_with":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[2] and other_triplet[1] == "affects":
                    if triplet[0] != other_triplet[2]:
                        co_occurs_triplets.append([triplet[0], "affects", other_triplet[2]])

    # 2. Rule of Prevention and Causation
    for triplet in kg_triplets:
        if triplet[1] == "prevents":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[2] and other_triplet[1] == "causes":
                    if triplet[0] != other_triplet[2]:
                        prevent_triplets.append([triplet[0], "prevents", other_triplet[2]])

    # 3. Rule of Treatment and Classification
    for triplet in kg_triplets:
        if triplet[1] == "treats":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[2] and other_triplet[1] == "is_a":
                    if triplet[0] != other_triplet[2]:
                        treatment_triplets.append([triplet[0], "treats", other_triplet[2]])

    # 4. Rule of Diagnosis and Interaction
    for triplet in kg_triplets:
        if triplet[1] == "diagnoses":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[0] and other_triplet[1] == "interacts_with":
                    if other_triplet[2] != triplet[0]:
                        diagnosis_triplets.append([other_triplet[2], "diagnoses", triplet[2]])

    # 5. Rule of Conjunction
    for triplet in kg_triplets:
        if triplet[1] == "co-occurs_with":
            for other_triplet in kg_triplets:
                if other_triplet[0] == triplet[0] and other_triplet[1] == "affects":
                    if triplet[2] != other_triplet[2]:
                        conjunction_triplets.append([triplet[2], "co-occurs_with", other_triplet[2]])

    # 6. Rule of Disjunction
    for triplet in kg_triplets:
        if triplet[1] == "prevents":
            X = triplet[0]
            Y = triplet[2]
            for other_triplet in kg_triplets:
                if other_triplet[1] == "causes" and other_triplet[0] == Y:
                    Z = other_triplet[2]
                    disjunction_triplets.append([X, "prevents", Z])
                    disjunction_triplets.append([X, "causes", Z])

    # Parse and remove duplicates
    return (
        parse_triple(remove_duplicate(co_occurs_triplets)),
        parse_triple(remove_duplicate(prevent_triplets)),
        parse_triple(remove_duplicate(treatment_triplets)),
        parse_triple(remove_duplicate(diagnosis_triplets)),
        parse_triple(remove_duplicate(conjunction_triplets)),
        parse_triple(remove_duplicate(disjunction_triplets)),
    )


def process_data(input_file, output_file):
    """
    Process data to include logical rules and save the results.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        url = item.get("url")
        caption = item.get("caption")
        finding = item.get("finding")
        other_info = item.get("other_info")
        image_name = item.get("image_name")
        kg_triplets = item.get("kg_triplets", [])

        # Apply logical rules
        r1, r2, r3, r4, r5, r6 = apply_rules_to_kg(kg_triplets)

        # Create processed sample
        processed_sample = {
            "url": url,
            "caption": caption,
            "finding": finding,
            "other_info": other_info,
            "image_name": image_name,
            "kg_triplets": kg_triplets,
            "rule_1": r1,
            "rule_2": r2,
            "rule_3": r3,
            "rule_4": r4,
            "rule_5": r5,
            "rule_6": r6,
        }
        processed_data.append(processed_sample)

    # Save the processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    print(f"Processed data saved to {output_file}")


# Input and output paths
input_file = "vqa_rad/processed_vqa_with_kg.json"
output_file = "vqa_rad/processed_vqa_with_rules.json"

# Process the data
process_data(input_file, output_file)
