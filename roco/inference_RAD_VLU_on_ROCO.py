
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

# Processor Setup
model_id = "../VQA-RAD/VLU_RAD"
processor = AutoProcessor.from_pretrained(model_id)

# BitsAndBytesConfig for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Model Loading
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
print("Model and processor loaded successfully.")


inference_prompt = """
You are an AI assistant for medical image analysis. 
Your task is to analyze the provided medical image, refine the initial knowledge graph (KG), and generate a detailed, concise, and medically accurate caption. 
The refined KG should guide the captioning process, ensuring the caption accurately describes the key observations, anatomical regions, and any relevant medical findings depicted in the image.

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
3. **Causal Reasoning**: Apply the logical rules to refine and enhance the initial KG.
4. **Alignment**: Ensure the refined KG is consistent with the visual information in the image and fully supports the given question and answer.

### Inputs:
- **Initial Knowledge Graph**: {initial_kg}
- **Initial Caption**: {caption}

### Task:
1. Refine the provided initial KG to correct any inaccuracies or misalignments.
2. Generate a structured, concise, and accurate refined KG aligned with the image and caption.
3. Ensure the refined KG serves as a reasoning tool for deriving the provided caption.

Dont generate anything else except refine KG in the given format.
### Refined Knowledge Graph:
- Rule of Co-occurrence: [head, relation, tail]
- Rule of Conjunction: [head, relation, tail]
- Rule of Diagnosis and Interaction: [head, relation, tail]
...

"""



# Dataset Paths
test_file = "roco data/test_with_kg.json"

# Function to load test samples
def load_test_samples(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

# Load test data
test_data = load_test_samples(test_file)
print(f"Loaded {len(test_data)} test samples.")

# Function for ROCO inference
def run_roco_inference(image_path, initial_kg, caption):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Create input text
    input_text = inference_prompt.format(initial_kg=json.dumps(initial_kg), caption=json.dumps(caption))
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

    # Generate output from model
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, temperature=0.2)

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
output_csv = "ROCO_inference_on VLU_RAD.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Caption", "initial kg", "Generated KG"])
    for i, sample in enumerate(test_data):
        image_path = sample["file_name"]
        initial_kg = sample["kg_triples"]  # Use kg_triples field as the initial KG
        caption = sample["caption"]

        print(f"\nSample {i + 1}:")
        print(f"Image Path: {image_path}")
        print(f"caption: {caption}")
        print(f"Initial Knowledge Graph: {initial_kg}")

        generated_response = run_roco_inference(image_path, initial_kg, caption)
        print(f"Generated Caption: {generated_response}")

        writer.writerow([image_path, caption, initial_kg, generated_response])
        if i == 100:  # Limit to 15 samples for testing
            break


