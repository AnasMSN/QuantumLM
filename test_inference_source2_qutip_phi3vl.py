import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from utils import find_highest_checkpoint
import random
import pandas as pd

# Globals for holding the loaded model and processor
MODEL = None
PROCESSOR = None


def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, PROCESSOR

    # If already loaded, just return
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR

    print("Loading base model...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        # quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'
    )
    # 2. Find highest checkpoint
    try:
        adapter_path = find_highest_checkpoint(checkpoint_root)
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except:
        model = base_model
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    MODEL = model
    PROCESSOR = processor
    return MODEL, PROCESSOR


def run_inference_phi3v(image: Image.Image, user_input: str, temperature: float = 0.0, 
                        max_tokens: int = 500, model_id: str = "microsoft/Phi-3-vision-128k-instruct") -> str:

    model, processor = initialize_model(model_id)
    
    # Construct messages for a typical Phi-3 style prompt
    messages = [
        {"role": "user", "content": f"<|image_1|>\n{user_input}"}
    ]
    # Tokenize prompt using the built-in chat template
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Prepare the model inputs
    inputs = processor(
        prompt,
        images=[image],
        return_tensors="pt"
    ).to("cuda")

    # Generation parameters
    generation_args = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": False
    }

    # Generate the output
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Remove input tokens from the output
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


if __name__ == "__main__":
    prompt = "I want you to act as a quantum computer specialized in performing Grover’s algorithm. I will type a circuit, and you will reply with what a quantum computer should output. I want you to only reply with the output in a dictionary that contains the top-30 probabilities and nothing else. Circuit:"


    FOLDER_PATH = 'Dataset/Source0/'
    FILE_PATTERN = os.path.join(FOLDER_PATH, 'quantum_circuits_*_qubit.csv')

    CSV_FILENAME = 'Dataset/Source1/quantum_circuits_3_qubit_test.csv'
    data = pd.read_csv(CSV_FILENAME)
    x_test = data['openqasm'][:]
    image_path_1 = data['image_path1'][:]
    image_path_2 = data['image_path2'][:]
    ground_truth = data['ground_truth'][:]

    for i in range(len(x_test)):
        # generate random number between 0 and 1 to chooose between image_path_1 and image_path_2, import random
        random_number = random.random()
        if random_number < 0.5:
            image = Image.open(image_path_1[i]).convert("RGB")
        else:
            image = Image.open(image_path_2[i]).convert("RGB")
            
        image = image.resize((336, 336))
        
        # image = Image.open().convert("RGB")
        user_input = prompt + x_test[i]
        temperature = 1.5
        max_tokens = 500
        model_id = "unsloth/Qwen2-VL-2B-Instruct"

        generated_text = run_inference_phi3v(image, user_input, temperature, max_tokens, model_id)
        # generate print for the number of the test case and the generated textn result of generated_text and ground_truth
        print(f"Test case {i+1}")
        print(f"Generated text: {generated_text}")
        print(f"Ground truth: {ground_truth[i]}")
        print("\n")