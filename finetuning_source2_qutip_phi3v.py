import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from transformers import TrainerCallback
from unsloth import FastVisionModel 
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import pandas as pd
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import sys
import time
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from PIL import Image
import re
from transformers import TrainerCallback
import shutil, os
from peft import PeftModel
import torch.nn as nn

class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, formatter):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.formatter = formatter
        self.placeholders = re.findall(r"{([^}]+)}", formatter)
        self.image_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_path = row['image']
        input_text = row['input']
        output_text = row['output']

        # Load and transform the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Unable to load image at path: {image_path}. Error: {e}")

        image = self.image_transform(image)

        data_dict = {}
        for placeholder in self.placeholders:
            if placeholder == 'prompt':
                data_dict[placeholder] = input_text
            elif placeholder == 'answer':
                data_dict[placeholder] = output_text
            else:
                raise ValueError(f"Unexpected placeholder '{placeholder}' in formatter.")

        # Format the text using the formatter
        try:
            formatted_text = self.formatter.format(**data_dict)
        except KeyError as e:
            raise KeyError(f"Missing key for formatter: {e}")

        # Tokenize the formatted text
        encodings = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )

        # Prepare labels by copying input_ids
        encodings['labels'] = encodings['input_ids'].clone()

        # Squeeze to remove the batch dimension
        encodings = {key: val.squeeze(0) for key, val in encodings.items()}

        # Add pixel_values to encodings
        encodings['pixel_values'] = image

        return encodings

class ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # Remove the unexpected 'num_items_in_batch' if present.
        kwargs.pop("num_items_in_batch", None)
        return self.model(*args, **kwargs)


class FinetunePhi3V:
    def __init__(self, 
                 data,  # New parameter to receive data directly
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="microsoft/Phi-3-vision-128k-instruct", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                ):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit loading
            bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation
            bnb_8bit_use_double_quant=True,  # Use double quantization for memory efficiency
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            _attn_implementation='eager',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=peft_r, 
            lora_alpha=peft_alpha, 
            lora_dropout=peft_dropout, 
            target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
            inference_mode = False
        )
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.formatter = "<|user|>\n<|image_1|>{prompt}<|end|><|assistant|>{answer}<|end|>"
        self.data = data  # Store the data

    def run(self):
        dataset = ImageTextDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            formatter=self.formatter
        )

        model = get_peft_model(self.base_model, self.peft_config)

        training_args = TrainingArguments(
            learning_rate=self.learning_rate,
            output_dir='./model_cp_phi3_qutip',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            bf16=True,
            dataloader_num_workers=0,
            report_to="none",
            optim=self.optim,
            logging_steps=1, 
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        

if __name__ == "__main__":
    moved_folder = 'Dataset/Source2/'
    CSV_FILENAME = moved_folder+ 'metadata-qutip-grayscale-updated.csv' 
    data = pd.read_csv(CSV_FILENAME)
    
    required_columns = ['type', 'image', 'ground_truth', 'prompt']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the CSV file.")
    
    # Shuffle the dataset with a controlled random state
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    train_data = train_data.dropna().reset_index(drop=True)


    x_train = train_data['type'][:]
    images = train_data['image'][:]
    y_train = train_data['ground_truth'][:]
    prompts = train_data['prompt'][:]

    fine_tune_data = []
    for i in range(len(x_train)):
        # print the index for debugging
        fine_tune_data.append({
            "image":  moved_folder + images[i],
            "input": prompts[i] + x_train[i],
            "output": y_train[i],
        })

    finetuner = FinetunePhi3V(
        data=fine_tune_data,
        epochs=10,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        gradient_accumulation_steps=8,
        # optim="adamw_torch_fused",
        # model_id="unsloth/Qwen2.5-VL-7B-Instruct",
        # model_id="unsloth/llava-v1.6-mistral-7b-hf",
        peft_r=32,
        peft_alpha=32,
        peft_dropout=0.0,
    )

    finetuner.run()
