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
base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model_path = "VLU_vqa_rad"  # Path to your fine-tuned VLU module

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
test_data_file = "VLU_vqa_rad_test_set.json"

# Define the refined inference prompt
inference_prompt = """
You are an AI assistant for medical image analysis. Your task is to analyze the provided medical image and transform the initial knowledge graph (KG) into a refined, well-structured, concise, and accurate knowledge graph. This new KG should guide in deriving an accurate answer and reasoning for the given question. Use the following logical rules to assist in your refinement process:

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
- **Question**: {question}
- **Answer**: {answer}

### Task:
1. Refine the provided initial KG to correct any inaccuracies or misalignments.
2. Generate a structured, concise, and accurate refined KG aligned with the image, question, and answer.
3. Ensure the refined KG serves as a reasoning tool for deriving the provided answer.

Dont generate anything else except refine KG in the given format.
### Refined Knowledge Graph:
- Rule of Co-occurrence: [head, relation, tail]
- Rule of Conjunction: [head, relation, tail]
- Rule of Diagnosis and Interaction: [head, relation, tail]
...

"""

# Load test samples from JSON
def load_test_samples(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

# Load test samples
test_data = load_test_samples(test_data_file)
print(f"Loaded {len(test_data)} test samples.")

# Function for VLU inference
# Function for VLU inference
def run_vlu_inference(image_path, question, answer, initial_kg):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Create the input prompt
    input_text = inference_prompt.format(
        question=question,
        answer=answer,
        initial_kg=json.dumps(initial_kg)
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
        output = model.generate(**inputs, max_new_tokens=300)

    # Decode the output and extract the response after "assistant"
    generated_response = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in generated_response:
        try:
            generated_kg = generated_response.split("assistant")[-1].strip()
            return generated_kg
        except IndexError:
            print(f"Malformed response: {generated_response}")
            return None
    else:
        return generated_response


# Save results to CSV
output_csv = "VLU_vqa_rad_inference_results.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Question", "Ground Truth Answer", "Initial Knowledge Graph", "Generated Knowledge Graph"])
    
    for i, sample in enumerate(test_data):
        image_path = sample["image"]
        question = sample["question"]
        ground_truth_answer = sample["answer"]
        initial_kg = sample["response"]  # Use the response field as the initial KG

        print(f"\nSample {i + 1}:")
        print(f"Image Path: {image_path}")
        print(f"Question: {question}")
        print(f"Ground Truth Answer: {ground_truth_answer}")
        print(f"Initial Knowledge Graph: {initial_kg}")

        generated_kg = run_vlu_inference(image_path, question, ground_truth_answer, initial_kg)
        print(f"Generated Knowledge Graph: {generated_kg}")

        writer.writerow([image_path, question, ground_truth_answer, initial_kg, generated_kg])
        # if i == 15:  # Limit to 5 samples for testing
        #     break

print(f"Inference results saved to {output_csv}")
