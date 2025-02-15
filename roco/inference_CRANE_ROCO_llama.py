
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
base_model_id = "VLU_ROCO_using_llama"
model_path = "CRANE_roco_on_llama"  # Path to your fine-tuned VLU module with rule
# model_path ="vqa_rad_model_withoutRule" # Path to your fine-tuned VLU module with rule

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

print("CRANE Model loaded successfully.")



# CRANE Inference Prompt
inference_prompt = """
You are an AI assistant for medical image analysis. Your task is to analyze the provided medical image and generate a caption based on visual observations and logical reasoning. Use the following logical rules to guide your analysis and reasoning process:

1. **Descriptive**: Clearly describes the observed findings in the image.
2. **Localized**: Mentions specific anatomical regions and directions.
3. **Causal**: Explains the possible cause-effect relationships based on the observations.

### Rules:
1. Rule of Co-occurrence: co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
2. Rule of Prevention and Causation: prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
3. Rule of Treatment and Classification: treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
4. Rule of Diagnosis and Interaction: diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
5. Rule of Conjunction: co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
6. Rule of Disjunction: (prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))
"""

# Test dataset path
test_file = "roco data/test_with_kg.json"

# Load the test dataset
def load_test_samples(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

test_data = load_test_samples(test_file)
print(f"Loaded {len(test_data)} test samples.")


# Inference Function
def run_inference(image_path, caption):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Create the input prompt
    input_text = inference_prompt ### + f"\n### Input Caption: {caption}\n"

    # Prepare input for the model
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
output_csv = "CRANE_roco_inference_results.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Input Caption", "Generated Caption"])
    
    for i, sample in enumerate(test_data):
        image_path = sample["file_name"]
        input_caption = sample["caption"]

        print(f"\nProcessing Sample {i + 1}/{len(test_data)}")
        print(f"Image Path: {image_path}")
        print(f"Input Caption: {input_caption}")

        generated_caption = run_inference(image_path, input_caption)
        print(f"Generated Caption: {generated_caption}")

        writer.writerow([image_path, input_caption, generated_caption])

        # Limit to 20 samples for testing purposes
        if i == 49:
            break

print(f"Inference results saved to {output_csv}")

