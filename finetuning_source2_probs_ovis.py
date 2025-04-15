import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import re
from torchvision import transforms
from transformers import TrainerCallback
import shutil, os
from peft import PeftModel
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ImageTextDataset(Dataset):
    def __init__(self, data, text_tokenizer, processor, formatter, visual_tokenizer):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.processor = processor

        self.text_tokenizer.padding_side = 'left'
        self.formatter = formatter
        self.visual_tokenizer = visual_tokenizer
        self.placeholders = re.findall(r"{([^}]+)}", formatter)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_path = row['image']
        input_text = row['input']
        output_text = row['output']

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Unable to load image at path: {image_path}. Error: {e}")

        # Format the text using the formatter
        data_dict = {p: input_text if p == 'prompt' else output_text for p in self.placeholders}
        formatted_text = self.formatter.format(**data_dict)

        encodings = self.text_tokenizer(
            text=formatted_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )

        # ✅ Use the tokenizer's expected processor
        pixel_values = self.visual_tokenizer.image_processor(images=image, return_tensors="pt")["pixel_values"]
        
        print("Transformed image tensor shape:", pixel_values.shape)
        
        encodings["pixel_values"] = pixel_values.squeeze(0)
        encodings["labels"] = encodings["input_ids"].clone()

        return {k: v.squeeze(0) if v.dim() == 2 else v for k, v in encodings.items()}

class FinetuneOvis2:
    def __init__(self, 
                 data,  # New parameter to receive data directly
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="AIDC-AI/Ovis2-16B", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                ):
        self.epochs = epochs
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit loading
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation
            bnb_8bit_use_double_quant=True  # Use double quantization for memory efficiency
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # _attn_implementation='eager',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=self.bnb_config,
            device_map="cuda:0"
        )
        
        self.visual_tokenizer = self.base_model.get_visual_tokenizer()
        self.visual_tokenizer.image_processor.size = {"height": 224, "width": 224}
        
        print("Expected image size:", self.visual_tokenizer.image_processor.size)
        print("Expected input format:", self.visual_tokenizer.image_processor.__class__)
        
        self.text_tokenizer = self.base_model.get_text_tokenizer()

        
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
            text_tokenizer=self.text_tokenizer,
            processor=self.processor,
            formatter=self.formatter,
            visual_tokenizer=self.visual_tokenizer
        )

        model = get_peft_model(self.base_model, self.peft_config).to("cuda:0")

        training_args = TrainingArguments(
            learning_rate=self.learning_rate,
            output_dir='./model_cp_ovis2',
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
    # picture_folder = 'generated_images_qutip_color_bigger_linspace_2d3d_blues/'
    CSV_FILENAME = 'metadata-qutip-color_2d3d_blues.csv' 
    data = pd.read_csv(CSV_FILENAME)
    
    required_columns = ['type', 'image', 'ground_truth', 'prompt']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the CSV file.")
    
    # Shuffle the dataset with a controlled random state
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    train_data = train_data.reset_index(drop=True)
    
    BEST_PROMPT = (
        "You are given a grayscale image representing a quantum optical state. "
        "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, etc.) "
        "as well as its key parameters (alpha/number of photons/density, number of qubits, and the linear space range). "
        "Please provide your answer in the format: "
        "\"This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\""
        "then extract your opinion on how you determine state, parameters, number of qubit from the image, "
    )


    x_train = train_data['type'][:]
    images = train_data['image'][:]
    y_train = train_data['ground_truth'][:]
    prompts = BEST_PROMPT

    fine_tune_data = []
    for i in range(len(x_train)):
        # print the index for debugging
        fine_tune_data.append({
            "image": images[i],
            "input": prompts + x_train[i],
            "output": y_train[i],
        })

    finetuner = FinetuneOvis2(
        data=fine_tune_data,
        epochs=10,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        gradient_accumulation_steps=16,
        # optim="adamw_torch_fused",
        model_id="AIDC-AI/Ovis2-8B",
        peft_r=16,
        peft_alpha=16,
        peft_dropout=0.0,
    )

    finetuner.run()

