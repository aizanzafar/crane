
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
base_model_id = "CRANE_VQA_RAD"
# model_path = "vqa_rad_model"  # Path to your fine-tuned VLU module with rule
# model_path ="vqa_rad_model_withoutRule" # Path to your fine-tuned VLU module with rule

# Load the processor and fine-tuned model
# BitsAndBytesConfig for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Model Loading
model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
print("Model and processor loaded successfully.")
processor = AutoProcessor.from_pretrained(base_model_id)
print("CRANE Model loaded successfully.")

# Path to test dataset JSON
test_data_file = "vqa_rad_test_set.json"

# ## Define the inference prompt without rule
inference_prompt = """
You are an AI assistant for medical image analysis. Your task is to analyze the provided medical image and answer the given question based on your observations and reasoning capabilities. Ensure your response includes:

1. **A precise and concise answer** to the question.
2. **A reasoning process** that supports your answer by referencing specific visual observations from the image and aligning with the question's context.

### Inputs:
- **Image**: Analyze the visual features in the provided medical image.
- **Question**: {question}

### Instructions:
1. Examine the medical image to identify key findings or abnormalities relevant to the question.
2. Provide a direct, accurate answer.
3. Justify your answer with a clear and concise reasoning process, referencing visual observations from the image.

### Expected Output:
1. **Answer**: Provide a precise answer.
2. **Reason**: Include a reasoning process that explains your observations and supports your answer.
"""


# Load test samples from JSON
def load_test_samples(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

# Load test samples
test_data = load_test_samples(test_data_file)
print(f"Loaded {len(test_data)} test samples.")

# Function for CRANE inference
def run_crane_inference(image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    input_text = inference_prompt.format(question=question)
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
# output_csv = "CRANE_inference_results_on_vqaRAD.csv"
# with open(output_csv, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Image Path", "Question", "ground truth", "Generated Answer"])
for i, sample in enumerate(test_data):
    image_path = sample["image"]
    question = sample["question"]
    ground_answer= sample["answer"]

    print(f"\nSample {i + 1}:")
    print(f"Image Path: {image_path}")
    print(f"Question: {question}")
    print(f"Ground Answer: {ground_answer}")

    generated_answer = run_crane_inference(image_path, question)

    print(f"Generated Answer: {generated_answer}")

    # writer.writerow([image_path, question,ground_answer, generated_answer])
    if i == 15:  # Limit for testing purposes
        break

# print(f"Inference results saved to {output_csv}")

