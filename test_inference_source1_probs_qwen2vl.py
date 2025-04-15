import os
import torch
from unsloth import FastVisionModel
from PIL import Image
import pandas as pd
import random  

# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None


def find_highest_checkpoint(checkpoint_dir: str) -> str:
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by the numeric portion after "checkpoint-"
    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    highest_checkpoint = checkpoints_sorted[-1]
    return os.path.join(checkpoint_dir, highest_checkpoint)


def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, TOKENIZER

    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    adapter_path = find_highest_checkpoint(checkpoint_root)
    print(f"Highest checkpoint found: {adapter_path}")
    
    print("Loading base model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name =  adapter_path,  # Trained model either locally or from huggingface
        load_in_4bit = False,
    )
    print("Base model loaded.")

    # 2. Find highest checkpoint

    MODEL = model.to("cuda")
    TOKENIZER = tokenizer

    return MODEL, TOKENIZER


def run_inference_qwenvl(image: Image.Image, user_input: str, temperature: float = 0.0, 
                        max_tokens: int = 500, model_id: str = "unsloth/Qwen2-VL-7B-Instruct") -> str:

    model, tokenizer = initialize_model(model_id)
    FastVisionModel.for_inference(model) 
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": user_input
                },
            ]
        }
    ]
    # Tokenize prompt using the built-in chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=0.1
    )
    generate_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return generated_text

if __name__ == "__main__":
    prompt = "from qiskit import QuantumCircuit\nfrom qiskit_aer import AerSimulator\nfrom qiskit_ibm_runtime import Sampler\nfrom qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager\ndef run_bell_state_simulator():\n    \"\"\" Define a phi plus bell state using Qiskit, transpile the circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and return the counts dictionary.\n"


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

        generated_text = run_inference_qwenvl(image, user_input, temperature, max_tokens, model_id)
        # generate print for the number of the test case and the generated textn result of generated_text and ground_truth
        print(f"Test case {i+1}")
        print(f"Generated text: {generated_text}")
        print(f"Ground truth: {ground_truth[i]}")
        print("\n")