import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel
import json
import csv

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and processor paths
base_model_id = "VLU_RAD"
model_path = "VLU_roco_on_VLU_RAD"  # Path to your fine-tuned VLU module

# Load the processor and fine-tuned model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the base model
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

# Load the fine-tuned weights into the base model
model = PeftModel.from_pretrained(base_model, model_path)
processor = AutoProcessor.from_pretrained(base_model_id)

print("VLU Module loaded successfully.")

# Path to test dataset JSON
test_file = "roco data/test_with_kg.json"

# Define the refined inference prompt
inference_prompt = """
You are an AI assistant for medical image analysis. 
Your task is to analyze the provided medical image and caption to transform the initial knowledge graph (KG) into a refined, well-structured, concise, and accurate knowledge graph That ensuring the caption accurately describes the key observations, anatomical regions, and any relevant medical findings depicted in the image. Use the following logical rules to assist in creating refined KG:

### Logical Rules:
1. **Rule of Co-occurrence**: co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
2. **Rule of Prevention and Causation**: prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
3. **Rule of Treatment and Classification**: treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
4. **Rule of Diagnosis and Interaction**: diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
5. **Rule of Conjunction**: co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
6. **Rule of Disjunction**: (prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))

### Instructions:
1. **Observation**: Examine the medical image to identify key findings or abnormalities.
2. **Localization**: Mention specific anatomical regions, directions, or relevant structures in the image.
3. **Causal Reasoning**: Apply the logical rules to refine the initial KG.
4. **Alignment**: Ensure the refined KG is consistent with the visual information in the image and fully supports the caption.
5. Apply the logical rules over Knowledge graph and generate new refine Knowledge graph
### Inputs:
- **Initial Knowledge Graph**: {initial_kg}
- **Initial Caption**: {caption}

### Task:
1. Refine the provided initial KG to correct any inaccuracies or misalignments.
2. New refine KG should tell which logical rules is applied and triplets generated by that.

NOTE:
Don't generate anything like comments or anything except refined KG in same format as inital kg was given.
"""

# Load the test dataset
def load_test_samples(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

test_data = load_test_samples(test_file)
print(f"Loaded {len(test_data)} test samples.")

# Function for inference
def run_inference(image_path, initial_kg, caption):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    input_text = inference_prompt.format(
        initial_kg=json.dumps(initial_kg[:60]),
        caption=caption
    )
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": input_text},
            {"type": "image", "image": image},
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # Generate output from the model
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, temperature=0.2)

    # Decode the output
    generated_response = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in generated_response:
        try:
            generated_ans = generated_response.split("assistant")[-1].strip()
            return generated_ans
        except IndexError:
            print(f"Malformed response: {generated_response}")
            return None
    else:
        return generated_response
# Save results to CSV
output_csv = "VLU_roco_inference_using_VLU_RAD_results.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Caption", "Initial KG", "Generated Response"])
    
    for i, sample in enumerate(test_data):
        image_path = sample["file_name"]
        initial_kg = sample["kg_triples"]
        caption = sample["caption"]

        print(f"\nProcessing Sample {i + 1}/{len(test_data)}")
        print(f"Image Path: {image_path}")
        print(f"Initial KG: {initial_kg}")
        print(f"Initial Caption: {caption}")

        generated_response = run_inference(image_path, initial_kg, caption)
        print(f"Generated Response: {generated_response}")

        writer.writerow([image_path, caption, initial_kg, generated_response])

        # Limit to 20 samples for testing purposes
        if i == 19:
            break

print(f"Inference results saved to {output_csv}")
