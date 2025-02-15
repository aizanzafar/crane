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

# Prompt for CRANE training
prompt = """
You are an AI assistant for medical image analysis. Your task is to analyze the provided medical image and answer the given question based on visual observations and logical reasoning. Use the following logical rules to guide your analysis and reasoning process:

### Rules:
1. **Rule of Co-occurrence**: co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
2. **Rule of Prevention and Causation**: prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
3. **Rule of Treatment and Classification**: treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
4. **Rule of Diagnosis and Interaction**: diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
5. **Rule of Conjunction**: co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
6. **Rule of Disjunction**: (prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))

### Instructions:
1. **Observation**: Carefully examine the medical image and identify key findings or abnormalities.
2. **Localization**: Determine specific anatomical regions or directions relevant to the question.
3. **Causal Reasoning**: Apply the logical rules to infer relationships and derive a meaningful answer.

### Question: {question}

Do not output anything else except the answer in a concise format.
"""

assistant_prompt = """
### Answer: {answer}
"""

# Dataset Paths
root_dir = "../../datasets/VQA-RAD"
image_dir = os.path.join(root_dir, "VQA_RAD Image Folder")
json_file = os.path.join(root_dir, "VQA_RAD Dataset Public.json")

# Function to load VQA-RAD dataset and split it into train, validation, and test
def load_vqa_rad_data(image_dir, json_file, split_ratios=(0.7, 0.15, 0.15)):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    dataset = []
    for entry in data:
        image_path = os.path.join(image_dir, entry["image_name"])
        if os.path.exists(image_path):
            # Add the main question-answer pair
            dataset.append({
                "image": image_path,
                "question": entry["question"],
                "answer": entry["answer"]
            })
            # Add the rephrased question-answer pair if available
            if "question_rephrase" in entry and entry["question_rephrase"] != "NULL":
                dataset.append({
                    "image": image_path,
                    "question": entry["question_rephrase"],
                    "answer": entry["answer"]
                })
    
    # Shuffle and split dataset
    random.shuffle(dataset)
    train_size = int(len(dataset) * split_ratios[0])
    val_size = int(len(dataset) * split_ratios[1])
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    print(f"Dataset split: Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


# Load the data
train_data, val_data, test_data = load_vqa_rad_data(image_dir, json_file)

# Save the test set for later inference
with open("vqa_rad_test_set.json", "w") as f:
    json.dump(test_data, f)

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
                    {"type": "text", "text": prompt.format(question=sample["question"])},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_prompt.format(answer=sample["answer"])}],
            },
        ]
    }

train_dataset = [format_data(sample) for sample in train_data if format_data(sample) is not None]
val_dataset = [format_data(sample) for sample in val_data if format_data(sample) is not None]

print(f"First sample from training dataset: {train_dataset[0]}")
print(f"Filtered Train samples: {len(train_dataset)}, Filtered Validation samples: {len(val_dataset)}")

# Processor Setup
model_id = "VLU module/finetune_vlu"
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
    output_dir="crane_vqa_rad_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=6,
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
processor.tokenizer.save_pretrained("crane_vqa_rad_model")
trainer.save_model("crane_vqa_rad_model")
processor.save_pretrained("crane_vqa_rad_model")
