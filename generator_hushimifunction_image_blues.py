import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from qutip import *
import time

# Create output directories
output_dir = "generated_hushimi_q_blues"
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
lin_space = [
    np.linspace(-lin_space5, lin_space5, 200),
    np.linspace(-lin_space6, lin_space6, 200),
    np.linspace(-lin_space7, lin_space7, 200),
    np.linspace(-lin_space8, lin_space8, 200),
    np.linspace(-lin_space9, lin_space9, 200),
    np.linspace(-lin_space10, lin_space10, 200)
]

# Data storage
data = []

# A helper function to plot and save 2D & 3D Wigner side-by-side
def plot_and_save_qfunc(rho, xvec, image_path):
    fig = plt.figure(figsize=(10, 5))

    # 2D plot (QuTiP Q function)
    ax1 = fig.add_subplot(1, 2, 1)
    Q = qfunc(rho, xvec, xvec)
    ax1.contourf(xvec, xvec, Q, 100, cmap='Blues')
    ax1.set_title("Q Function 2D")
    ax1.set_xlabel("Re(α)")
    ax1.set_ylabel("Im(α)")

    # 3D plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(xvec, xvec)
    surf = ax2.plot_surface(X, Y, Q, cmap='Blues', edgecolor='black', linewidth=0.25, antialiased=True)
    ax2.set_title("Q Function 3D")
    ax2.set_xlabel("Re(α)")
    ax2.set_ylabel("Im(α)")
    ax2.set_zlabel("Q")

    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close(fig)

# Generate images
counter = 0

####################################################
# 1) Cat state
####################################################
for N in N_values:
    for alpha in alpha_values:
        state_name = "Cat state"
        try:
            rho_cat = (coherent(N, alpha) + coherent(N, -alpha)).unit()
            
            # loop through each lin_space
            for idx, xvec in enumerate(lin_space):
                W = qfunc(rho_cat, xvec, xvec)
                
                image_path = os.path.join(
                    output_dir,
                    f"{state_name.replace(' ', '_')}_N{N}_alpha{alpha}_linspace{lin_space_idx[idx]}.png"
                )
                
                # Plot 2D & 3D side by side
                plot_and_save_qfunc(rho_cat, xvec, image_path)
                
                ground_truth = (
                    f"This is a {state_name.lower()} with alpha={alpha}, "
                    f"number of qubits={N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                )
                data.append([image_path, state_name, ground_truth])
                
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, alpha={alpha}): {e}")
            
        counter += 1
        if counter % 10 == 0:
            time.sleep(1)

####################################################
# 2) Coherent state
####################################################
for N in N_values:
    for alpha in alpha_values:
        state_name = "Coherent state"
        try:
            rho_coherent = coherent_dm(N, alpha)
            
            for idx, xvec in enumerate(lin_space):
                W = qfunc(rho_coherent, xvec, xvec)
                
                image_path = os.path.join(
                    output_dir,
                    f"{state_name.replace(' ', '_')}_N{N}_alpha{alpha}_linspace{lin_space_idx[idx]}.png"
                )
                
                plot_and_save_qfunc(rho_coherent, xvec, image_path)
                
                ground_truth = (
                    f"This is a {state_name.lower()} with alpha={alpha}, "
                    f"number of qubits={N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                )
                data.append([image_path, state_name, ground_truth])
                
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, alpha={alpha}): {e}")
            
        counter += 1
        if counter % 10 == 0:
            time.sleep(1)

####################################################
# 3) Thermal state
####################################################
for N in N_values:
    for n in n_values:
        rho_thermal = thermal_dm(N, n)
        state_name = "Thermal state"
        
        try:
            for idx, xvec in enumerate(lin_space):
                W = qfunc(rho_thermal, xvec, xvec)
                
                image_path = os.path.join(
                    output_dir,
                    f"{state_name.replace(' ', '_')}_N{N}_photons{n}_linspace{lin_space_idx[idx]}.png"
                )
                
                plot_and_save_qfunc(rho_thermal, xvec, image_path)
                
                ground_truth = (
                    f"This is a {state_name.lower()} with average photon={n}, "
                    f"number of qubits={N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                )
                data.append([image_path, state_name, ground_truth])
            
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, n={n}): {e}")
            
        counter += 1
        if counter % 10 == 0:
            time.sleep(1)

####################################################
# 4) Fock state
####################################################
for N in N_values:
    for n in n_values:
        if n < N:
            rho_fock = fock_dm(N, n)
            state_name = "Fock state"
            
            try:
                for idx, xvec in enumerate(lin_space):
                    W = qfunc(rho_fock, xvec, xvec)
                    
                    image_path = os.path.join(
                        output_dir,
                        f"{state_name.replace(' ', '_')}_N{N}_photons{n}_linspace{lin_space_idx[idx]}.png"
                    )
                    
                    plot_and_save_qfunc(rho_fock, xvec, image_path)
                    
                    ground_truth = (
                        f"This is a {state_name.lower()} with photon={n}, "
                        f"number of qubits={N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                    )
                    data.append([image_path, state_name, ground_truth])
            except Exception as e:
                print(f"Error generating {state_name} state (N={N}, n={n}): {e}")
                
            counter += 1
            if counter % 10 == 0:
                time.sleep(1)

####################################################
# 5) Random state
####################################################
for N in N_values:
    for density in density_values:
        rho_random = rand_dm(N, density)
        state_name = "Random state"
        
        try:
            for idx, xvec in enumerate(lin_space):
                W = qfunc(rho_random, xvec, xvec)
                
                image_path = os.path.join(
                    output_dir,
                    f"{state_name.replace(' ', '_')}_N{N}_density{density}_linspace{lin_space_idx[idx]}.png"
                )
                
                plot_and_save_qfunc(rho_random, xvec, image_path)
                
                ground_truth = (
                    f"This is a {state_name.lower()} with density={density}, "
                    f"number of qubits={N} in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
                )
                data.append([image_path, state_name, ground_truth])
        except Exception as e:
            print(f"Error generating {state_name} state (N={N}, density={density}): {e}")
            
        counter += 1
        if counter % 10 == 0:
            time.sleep(1)

####################################################
# 6) Number state
####################################################
for N in N_values:
    rho_num = num(N)
    state_name = "Number state"
    
    try:
        for idx, xvec in enumerate(lin_space):
            W = qfunc(rho_num, xvec, xvec)
            
            image_path = os.path.join(
                output_dir,
                f"{state_name.replace(' ', '_')}_N{N}_linspace{lin_space_idx[idx]}.png"
            )
            
            plot_and_save_qfunc(rho_num, xvec, image_path)
            
            ground_truth = (
                f"This is a {state_name.lower()} with number of qubits={N} "
                f"in the linear space -{lin_space_idx[idx]} to {lin_space_idx[idx]}."
            )
            data.append([image_path, state_name, ground_truth])
    except Exception as e:
        print(f"Error generating {state_name} state (N={N}): {e}")

# Save metadata to CSV
df = pd.DataFrame(data, columns=["image", "type", "ground_truth"])
df.to_csv("metadata-hushimiq-color_2d3d_blues.csv", index=False)

print("Image generation (with side-by-side 2D & 3D Hushimi plots) and metadata saving completed.")
