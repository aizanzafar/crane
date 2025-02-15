import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
import json
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Updated Prompt for VLU Training
prompt = """
You are an AI assistant for medical image analysis. 
Your task is to analyze the provided medical image and transform the initial knowledge graph (KG) into a refined, well-structured, concise, and accurate knowledge graph.
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
4. **Alignment**: Ensure the refined KG is consistent with the visual information in the image and fully supports the caption.

### Inputs:
- **Initial Knowledge Graph**: {initial_kg}
- **Initial Caption**: {caption}

### Task:
1. Refine the provided initial KG to correct any inaccuracies or misalignments.
2. Generate a structured, concise, and accurate refined KG aligned with the image and initial caption.
3. Create a new, detailed, and accurate caption for the image, supported by the refined KG and logical reasoning.

"""

# Dataset Paths
train_file = "roco data/train_with_kg.json"
val_file = "roco data/validation_with_kg.json"

# Function to load the ROCO dataset
def load_roco_data(file_path, min_caption_tokens=15):
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = []
    for entry in data:
        caption_tokens = entry["caption"].split()
        if len(caption_tokens) > min_caption_tokens:
            image_path = entry["file_name"]
            if os.path.exists(image_path):
                dataset.append({
                    "image": image_path,
                    "caption": entry["caption"],
                    "initial_kg": entry["kg_triples"],
                })
    
    print(f"Loaded {len(dataset)} samples from {file_path} with captions > {min_caption_tokens} tokens")
    return dataset

# Load the data
train_data = load_roco_data(train_file, min_caption_tokens=15)
val_data = load_roco_data(val_file)

# Format dataset for training
def format_data(sample):
    try:
        image = Image.open(sample["image"]).convert("RGB")
    except Exception as e:
        print(f"Error opening image {sample['image']}: {e}")
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.format(
                        caption=sample["caption"],
                        initial_kg=json.dumps(sample["initial_kg"][:60])
                    )},
                    {"type": "image", "image": image},
                ],
            },
        ]
    }

train_dataset = [format_data(sample) for sample in train_data if format_data(sample) is not None][:17000]
val_dataset = [format_data(sample) for sample in val_data if format_data(sample) is not None][:3500]

print(f"First three samples from training dataset: {train_dataset[:3]}")
print(f"Filtered Train samples: {len(train_dataset)}, Filtered Validation samples: {len(val_dataset)}")

# Processor Setup
model_id = "VLU_RAD"
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


# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Training Configuration
args = SFTConfig(
    output_dir="VLU_roco_on_VLU_RAD",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": True},
    dataset_kwargs={"skip_prepare_dataset": True},
)
args.remove_unused_columns = False

# Data Collator
def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

# Trainer Initialization
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    dataset_text_field="",  # Placeholder for dataset field
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

# Train the model
trainer.train()

# Save the final model
processor.tokenizer.save_pretrained("VLU_roco_on_VLU_RAD")
trainer.save_model("VLU_roco_on_VLU_RAD")
processor.save_pretrained("VLU_roco_on_VLU_RAD")
