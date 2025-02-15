import os
import json

# Define dataset root directories
root_dir = "roco-dataset-master/data"
splits = ["train", "validation", "test"]

# Define subdirectories
categories = ["radiology"]

# Function to parse a text file into a dictionary or list
def parse_text_file(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        lines = f.readlines()
    return {line.split()[0]: " ".join(line.split()[1:]) for line in lines}

def create_json(split):
    dataset = []
    for category in categories:
        category_dir = os.path.join(root_dir, split, category)
        images_dir = os.path.join(category_dir, "images")

        # File paths
        caption_file = os.path.join(category_dir, "captions.txt")
        keywords_file = os.path.join(category_dir, "keywords.txt")
        cuis_file = os.path.join(category_dir, "cuis.txt")
        semtypes_file = os.path.join(category_dir, "semtypes.txt")

        # Parse files
        captions = parse_text_file(caption_file)
        keywords = parse_text_file(keywords_file)
        cuis = parse_text_file(cuis_file)
        semtypes = parse_text_file(semtypes_file)

        # Iterate through images
        for image_id in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_id)
            file_id = os.path.splitext(image_id)[0]

            dataset.append({
                "id": file_id,
                "file_name": image_path,
                "caption": captions.get(file_id, ""),
                "keywords": keywords.get(file_id, "").split(", "),
                "cuis": cuis.get(file_id, "").split(", "),
                "semtypes": semtypes.get(file_id, "").split(", ")
            })
    return dataset

output_dir="roco data"
# Generate JSON files
for split in splits:
    print(f"Processing {split}...")
    json_data = create_json(split)
    output_file = os.path.join(output_dir, f"{split}.json")
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved {split}.json with {len(json_data)} entries.")
