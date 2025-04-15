from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

class OvisDataset(Dataset):
    def __init__(self, image_text_data, tokenizer, model):
        self.data = image_text_data
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, question, answer = self.data[idx]
        print(f"self.data: {self.data[idx]}")
        print(f"Processing image: {image_path}, question: {question}, answer: {answer}")
        image = Image.open(image_path).convert("RGB")
        query = f"<image>\n{question}"
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image], max_partition=9)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # This is causal LM, label is same as input
        }

if __name__  == "__main__":  
    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis2-4B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()


    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False
    )

    model = get_peft_model(model, peft_config)
    model = model.to_empty(device="cuda:0")  # Manually move from meta

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
            "image_path": images[i],
            "question": prompts + x_train[i],
            "answer": y_train[i],
        })

    train_dataset = OvisDataset(fine_tune_data, text_tokenizer, model)

    training_args = TrainingArguments(
        output_dir="./ovis2_34b_lora",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        bf16=True,
        learning_rate=2e-5,
        gradient_accumulation_steps=4,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()


