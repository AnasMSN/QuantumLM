from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-14B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,    # Supports any, but = 0 is optimized
    bias = "none",       # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts, }

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Fixed major bug in latest Unsloth
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit", # Save more memory
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

tokenizer.decode(trainer.train_dataset[5]["input_ids"])

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# messages = [
#     {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
# ).to("cuda")

# outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
#                          temperature = 1.5, min_p = 0.1)
# tokenizer.batch_decode(outputs)

# FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# messages = [
#     {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize = True,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
# ).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")

# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


#### sample question Paris tower
# from unsloth import FastLanguageModel
# from transformers import TextStreamer

# if False:
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name="lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#     )

# FastLanguageModel.for_inference(model)

# messages = [
#     {"role": "user", "content": "Describe a tall tower in the capital of France."},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True, # Must add for generation
#     return_tensors="pt",
# ).to("cuda")

# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)


#### sample question NY statue
# from unsloth import FastLanguageModel
# from transformers import TextStreamer

# if False:    
#     new_model, new_tokenizer = FastLanguageModel.from_pretrained(
#         model_name="lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#     )
#     FastLanguageModel.for_inference(new_model) # Enable native 2x faster inference
    
#     messages = [
#         {"role": "user", "content": "Describe a famous statue in New York City."},
#     ]
#     inputs = new_tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True, # Must add for generation
#         return_tensors="pt",
#     ).to("cuda")
    
#     text_streamer = TextStreamer(new_tokenizer, skip_prompt = True)
#     _ = new_model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)


#### sample question Myanmar
from unsloth import FastLanguageModel
from transformers import TextStreamer

if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "Describe a famoous place in Myanmar."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True, # Must add for generation
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)0