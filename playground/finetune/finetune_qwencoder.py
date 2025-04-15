import psutil
import torch
import os, time, shutil
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bf16_supported
from unsloth.chat_templates import get_chat_template  # Import get_chat_template
from datasets import Dataset  # Import Dataset from Hugging Face datasets

# https://huggingface.co/unsloth
class FinetuneLM:
    def __init__(
        self,
        data, 
        epochs=1,
        learning_rate=5e-5, #0.00005
        warmup_ratio=0.1,
        gradient_accumulation_steps=16, #batch_size
        optim="adamw_torch",
        model_id="unsloth/Phi-3.5-mini-instruct",
        peft_r=8,
        peft_alpha=16,
        peft_dropout=0.0,
    ):
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
            
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=False,
            use_gradient_checkpointing=False,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.base_model,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            r=peft_r,
            lora_alpha=peft_alpha,
            lora_dropout=peft_dropout,
            bias="none",
            use_rslora=False,
            loftq_config=None
        )

    def format_data(self, row):
        user_prompt = row["input"]
        assistant_answer = row["output"]
        formatted_text = (
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_answer}<|im_end|>"
        )
        return {"text": formatted_text}
    
    def format_data_chatml(self, row, tokn):
        user_prompt = row["input"]
        assistant_answer = row["output"]
        row_json = [
            {"role": "user", "content": f"{row['input']}"},
            {"role": "assistant", "content": row["output"]}
        ]
        try:
            formatted_text = tokn.apply_chat_template(row_json, tokenize=False, add_generation_prompt = False)
        except Exception as e:
            formatted_text = (
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_answer}<|im_end|>"
            )
        return {"text": formatted_text}


    def run(self):
        if "mistral" in self.model_id.lower():
            template_name = "mistral"
        elif "llama" in self.model_id.lower():
            template_name = "llama-3"
        elif "deepseek" in self.model_id.lower() and "qwen" in self.model_id.lower():
            template_name = "chatml"
        else:
            template_name = None

        if template_name is not None:
            tokn = get_chat_template(
                self.tokenizer,
                chat_template=template_name,
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
                map_eos_token=True,
            )
            formatted_data = [self.format_data_chatml(row, tokn) for row in self.data]
        else:
            # Otherwise, default to your simpler Qwen/Phi style
            formatted_data = [self.format_data(row) for row in self.data]

        dataset = Dataset.from_list(formatted_data)

        # Create SFT config
        training_args = SFTConfig(
            learning_rate=self.learning_rate,
            output_dir='./model_cp',
            optim=self.optim,
            logging_steps=1,
            report_to="none",
            fp16=(not is_bf16_supported()),
            bf16=is_bf16_supported(),
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_strategy="steps",
            max_seq_length=2048,
        )

        # Prepare model for training
        FastLanguageModel.for_training(self.model)

        # Initialize and run trainer
        trainer = SFTTrainer(
            model=self.model,
            # dataset_text_field="text",
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        trainer.train()
        

sample_data = [{"input": "your input", "output": "ground truth"},]

model_id = "unsloth/Qwen2.5-Coder-7B-Instruct"
finetuner = FinetuneLM(data=sample_data, epochs=1, learning_rate=5e-6, model_id=model_id, peft_alpha=16, \
        peft_r=16, peft_dropout=0.0, gradient_accumulation_steps=8, warmup_ratio=0.1)
finetuner.run()