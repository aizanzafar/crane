import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and processor paths
base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model_path = "VLU_roco_on_llama"  # Path to your fine-tuned VLU module

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

# Merge LoRA weights and unload PEFT
print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

# Save the model and processor
output_dir = "VLU_ROCO_using_llama"
os.makedirs(output_dir, exist_ok=True)

print("Saving the model and processor...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("Model and processor saved successfully.")
