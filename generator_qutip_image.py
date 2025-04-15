import numpy as np
import pandas as pd
import os
from qutip import *
from matplotlib import pyplot as plt
import time

# Create output directories
output_dir = "generated_images_qutip_color_bigger_linspace"
os.makedirs(output_dir, exist_ok=True)

# Parameters
N_values = [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
alpha_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
n_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
density_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Define linear spaces
lin_space5 = 5
lin_space6 = 6
lin_space7 = 7
lin_space8 = 8
lin_space9 = 9
lin_space10 = 10
lin_space_idx = [lin_space5, lin_space6, lin_space7, lin_space8, lin_space9, lin_space10]
lin_space = [np.linspace(-lin_space5, lin_space5,200), np.linspace(-lin_space6, lin_space6,200), np.linspace(-lin_space7, lin_space7,200), np.linspace(-lin_space8, lin_space8,200), np.linspace(-lin_space9, lin_space9,200), np.linspace(-lin_space10,lin_space10,200)]
xvec = lin_space[0]

# Data storage
data = []

# Generate images
counter = 0
for N in N_values:
    for alpha in alpha_values:
        state_name = "Cat state"
        try:
            rho_cat = (coherent(N, alpha) + coherent(N, -alpha)).unit()
            
            # loop lin_space based with index and value
            for idx, xvec in enumerate(lin_space):
                W = wigner(rho_cat, xvec, xvec)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                cont = ax.contourf(xvec, xvec, W, 100, cmap='viridis')
                
                image_path = os.path.join(output_dir, f"{state_name.replace(' ', '_')}_N{N}_alpha{alpha}_linspace{lin_space_idx[idx]}.png")
                plt.savefig(image_path)
                plt.close(fig)
                
                ground_truth = f"This is a {state_name.lower()} with alpha equal to {alpha}, number of qubits equal to {N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                data.append([image_path, state_name, "", ground_truth])
                
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, alpha={alpha}): {e}")
            
        counter += 1
        
        if counter % 10 == 0:
            # sleep execution for 1 second
            time.sleep(1)
            
for N in N_values:
    for alpha in alpha_values:
        state_name = "Coherent state"
        try:
            rho_coherent = coherent_dm(N, alpha)
            
            # loop lin_space based with index and value
            for idx, xvec in enumerate(lin_space):
                W = wigner(rho_coherent, xvec, xvec)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                cont = ax.contourf(xvec, xvec, W, 100, cmap='viridis')
                
                image_path = os.path.join(output_dir, f"{state_name.replace(' ', '_')}_N{N}_alpha{alpha}_linspace{lin_space_idx[idx]}.png")
                plt.savefig(image_path)
                plt.close(fig)
                
                ground_truth = f"This is a {state_name.lower()} with alpha equal to {alpha}, number of qubits equal to {N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                data.append([image_path, state_name, "", ground_truth])
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, alpha={alpha}): {e}")
            
        counter += 1
            
        if counter % 10 == 0:
            # sleep execution for 1 second
            time.sleep(1)
            
for N in N_values:
    for n in n_values:
        rho_thermal = thermal_dm(N, n)
        
        state_name = "Thermal state"
        
        try:
            # loop lin_space based with index and value
            for idx, xvec in enumerate(lin_space):
                W = wigner(rho_thermal, xvec, xvec)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                cont = ax.contourf(xvec, xvec, W, 100, cmap='viridis')
                
                image_path = os.path.join(output_dir, f"{state_name.replace(' ', '_')}_N{N}_photons{n}_linspace{lin_space_idx[idx]}.png")
                plt.savefig(image_path)
                plt.close(fig)
                
                ground_truth = f"This is a {state_name.lower()} with average number of photons is {n}, number of qubits equal to {N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                data.append([image_path, state_name, "", ground_truth])
            
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, n={n}): {e}")
            
        counter += 1
        
        if counter % 10 == 0:
            # sleep execution for 1 second
            time.sleep(1)
            
for N in N_values:
    for n in n_values:
        if n < N:
            rho_fock = fock_dm(N, n)
            
            state_name = "Fock state"
            
            try:
                # loop lin_space based with index and value
                for idx, xvec in enumerate(lin_space):
                    W = wigner(rho_fock, xvec, xvec)
                    
                    fig, ax = plt.subplots(figsize=(4, 4))
                    cont = ax.contourf(xvec, xvec, W, 100, cmap='viridis')
                    
                    image_path = os.path.join(output_dir, f"{state_name.replace(' ', '_')}_N{N}_photons{n}_linspace{lin_space_idx[idx]}.png")
                    plt.savefig(image_path)
                    plt.close(fig)
                    
                    ground_truth = f"This is a {state_name.lower()} with average number of photons is {n}, number of qubits equal to {N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                    data.append([image_path, state_name, "", ground_truth])
            except Exception as e:
                print(f"Error generating {state_name} state (N={N}, n={n}): {e}")
                
            counter += 1
            
            if counter % 10 == 0:
                # sleep execution for 1 second
                time.sleep(1)
            
for N in N_values:
    for density in density_values:
        rho_random = rand_dm(N, density)
        
        state_name = "Random state"
        
        try:
            # loop lin_space based with index and value
            for idx, xvec in enumerate(lin_space):
                W = wigner(rho_random, xvec, xvec)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                cont = ax.contourf(xvec, xvec, W, 100, cmap='viridis')
                
                image_path = os.path.join(output_dir, f"{state_name.replace(' ', '_')}_N{N}_density{density}_linspace{lin_space_idx[idx]}.png")
                plt.savefig(image_path)
                plt.close(fig)
                
                ground_truth = f"This is a {state_name.lower()} with density is {density}, number of qubits equal to {N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                data.append([image_path, state_name, "", ground_truth])
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, density={density}): {e}")
            
        counter += 1
        
        if counter % 10 == 0:
            # sleep execution for 1 second
            time.sleep(1)
            
for N in N_values:
    rho_num = num(N)
    
    state_name = "Number state"
    
    try:
        # loop lin_space based with index and value
        for idx, xvec in enumerate(lin_space):
            W = wigner(rho_num, xvec, xvec)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            cont = ax.contourf(xvec, xvec, W, 100, cmap='viridis')
            
            image_path = os.path.join(output_dir, f"{state_name.replace(' ', '_')}_N{N}_linspace{lin_space_idx[idx]}.png")
            plt.savefig(image_path)
            plt.close(fig)
            # from here, add <think> and </think> to implement reasoning model
            ground_truth = f"This is a {state_name.lower()} with number of qubits equal to {N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
            data.append([image_path, state_name, "", ground_truth])
    except Exception as e:
        print(f"Error generating {state_name} state (N={N}): {e}")
        


# Save metadata to CSV
df = pd.DataFrame(data, columns=["image", "type", "prompt", "ground_truth"])
df.to_csv("metadata-qutip-color.csv", index=False)

print("Image generation and metadata saving completed.")
