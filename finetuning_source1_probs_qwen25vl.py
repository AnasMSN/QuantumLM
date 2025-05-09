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
import glob
import os

class FinetuneQwenVL:
    def __init__(self, 
                 data,
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="unsloth/Qwen2-VL-7B-Instruct", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                ):
        self.epochs = epochs
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        self.base_model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name = self.model_id,
            load_in_4bit = False,
            use_gradient_checkpointing = "unsloth",
        )
        self.model = FastVisionModel.get_peft_model(
            self.base_model,
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers
            r = peft_r,           
            lora_alpha = peft_alpha,  
            lora_dropout = peft_dropout,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  
            loftq_config = None
        )
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.data = data
    
    def format_data(self, row):
        image_path = row["image"]
        input_text = row['input']
        output_text = row['output']
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((336, 336))
        except Exception as e:
            raise FileNotFoundError(f"Unable to load image at path: {image_path}. Error: {e}")

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text,
                        },
                        {
                            "type": "image",
                            "image": image,  
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": output_text,
                        }
                    ],
                },
            ],
        }

    def run(self):
        """
        Executes the fine-tuning process.
        """
        converted_dataset = [self.format_data(row) for row in self.data]
        converted_dataset = converted_dataset
        training_args = SFTConfig(
            learning_rate=self.learning_rate,
            output_dir='./model_cp',
            optim=self.optim,
            logging_steps=1,
            report_to="none",
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            weight_decay = 0.01,            # Regularization term for preventing overfitting
            lr_scheduler_type = "linear",   # Chooses a linear learning rate decay
            seed = 3407,
            logging_strategy = "steps",
            # load_best_model_at_end = True,
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        )
        FastVisionModel.for_training(self.model)
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            data_collator = UnslothVisionDataCollator(self.model, self.tokenizer), # Must use!
            train_dataset = converted_dataset,
            args = training_args,
        )
        trainer.train()
        

if __name__ == "__main__":
    # Define the folder path and pattern for files
    FOLDER_PATH = 'Dataset/Source0/'
    FILE_PATTERN = os.path.join(FOLDER_PATH, 'quantum_circuits_*_qubit.csv')

    # Get all matching CSV files
    csv_files = glob.glob(FILE_PATTERN)
    csv_files
    print(csv_files)

    # Initialize list to store processed data
    data_list = []
    prompt = "I want you to act as a quantum computer specialized in performing Grover’s algorithm. I will type a circuit, and you will reply with what a quantum computer should output. I want you to only reply with the output in a dictionary that contains the top-30 probabilities and nothing else. Circuit:"


    # Iterate over all CSV files
    for file in csv_files:
        print(file)
        df = pd.read_csv(file)

        # Iterate through each row
        for _, row in df.iterrows():
            # Fetch image paths
            image1 = row['image_path1']
            image2 = row['image_path2']

            # Create entries linking each image path with each programming language representation
            for lang in ['openqasm', 'qiskit', 'quil', 'cirq', 'qobj']:
                data_list.append({
                    'image': image1,
                    'input': prompt + row[lang],
                    'ground_truth': row['ground_truth']
                })
                data_list.append({
                    'image_path': image2,
                    'input': prompt + row[lang],
                    'ground_truth': row['ground_truth']
                })
    
    # CSV_FILENAME = 'Dataset/Source1/quantum_circuits_3_qubit.csv' 
    # data = pd.read_csv(CSV_FILENAME)
    # x_train = data['openqasm'][:]
    # image_path_1 = data['image_path1'][:]
    # image_path_2 = data['image_path2'][:]
    # y_train = data['ground_truth'][:]

    # data = []
    

    # for i in range(len(x_train)):
    #     data.append({
    #         "image": image_path_1[i],
    #         "input": prompt + x_train[i],
    #         "output": y_train[i],
    #     })

    finetuner = FinetuneQwenVL(
        data=data_list,
        epochs=40,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=8,
        optim="adamw_torch_fused",
        model_id="unsloth/Qwen2.5-VL-7B-Instruct",
        peft_r=32,
        peft_alpha=32,
        peft_dropout=0.05,
    )

    finetuner.run()
