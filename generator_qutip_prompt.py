import pandas as pd
import random

# List of 100 prompts
qutip_prompts = [
    "What quantum state is depicted in this image?",
    "Identify the type of quantum state based on this visualization.",
    "Determine whether this image represents a coherent or thermal state.",
    "Analyze the quantum state shown in this Wigner function plot.",
    "What is the photon number distribution in this quantum state?",
    "Identify the value of alpha from this quantum state visualization.",
    "Based on this image, what is the mean photon number?",
    "Determine the Hilbert space dimension from this quantum state.",
    "Is this a pure or mixed quantum state? Explain your answer.",
    "Estimate the average number of photons based on this visualization.",
    "Identify whether this image corresponds to a cat state or a coherent state.",
    "Is this state characterized by quantum superposition?",
    "Does this image correspond to a number state with N = 10?",
    "What is the density matrix representation of the quantum state in the image?",
    "Based on this image, is the state entangled?",
    "Compare this state with a Fock state. Are they similar?",
    "Identify the quantum state with the highest density value in this dataset.",
    "Is the quantum state depicted here highly nonclassical?",
    "Identify the linear space range used to generate this visualization.",
    "Does this image correspond to a Wigner function of a squeezed state?",
    "Compare this visualization with a thermal state representation.",
    "Determine the Wigner negativity of the state in this image.",
    "How does alpha affect the quantum state shown in this image?",
    "Identify whether this state has a large or small displacement parameter.",
    "Compare this image to a vacuum state Wigner function.",
    "Identify whether this state is a superposition of coherent states.",
    "What is the most probable number state component in this image?",
    "Explain the phase space distribution of the quantum state.",
    "Identify whether this quantum state is closer to a classical or quantum description.",
    "Is the state depicted in this image highly sensitive to decoherence?",
    "Does this image show a thermal state with n = 5?",
    "Identify the rank of the density matrix used to generate this image.",
    "Explain the quadrature variance in this quantum state.",
    "What is the most dominant quantum feature in this Wigner function?",
    "Identify whether this state can be used for quantum error correction.",
    "Determine the purity of this quantum state from its visualization.",
    "Identify the type of quantum state transformation applied in this image.",
    "Compare the density matrix of this state to a maximally mixed state.",
    "What does the symmetry of this quantum state imply?",
    "Identify the state's parity from this visualization.",
    "Is this a displaced Fock state? Justify your answer.",
    "Determine whether this is a non-Gaussian quantum state.",
    "Identify the main characteristics of this quantum state.",
    "Is this an eigenstate of the number operator?",
    "Determine if this image represents a squeezed vacuum state.",
    "Compare this state to an entangled bipartite state.",
    "Identify the decoherence effects in this Wigner function representation.",
    "Estimate the Wigner negativity of the state in this image.",
    "Identify the region of highest probability in this quantum state visualization.",
    "Compare this state's Wigner function to that of a coherent state.",
    "Identify the rotation of this quantum state in phase space.",
    "What is the effect of increasing alpha in this state?",
    "Identify whether this state has a phase-space symmetry.",
    "Compare this image to a mixture of coherent states.",
    "Identify whether this quantum state corresponds to a finite-dimensional Hilbert space.",
    "Does this Wigner function have a negative region? If so, where?",
    "What does the central peak in this Wigner function indicate?",
    "Identify whether this state is robust to noise.",
    "Compare this quantum state to a single-photon state.",
    "Is this image generated from a density matrix with a high rank?",
    "Identify whether this quantum state is phase-squeezed or amplitude-squeezed.",
    "What is the primary nonclassical feature of this Wigner function?",
    "Does this state resemble a coherent state with large amplitude?",
    "Identify whether this state is a displaced thermal state.",
    "Compare this Wigner function to that of an optical cat state.",
    "Identify the main peak location in this quantum state visualization.",
    "How does increasing N affect the shape of this Wigner function?",
    "Identify whether this quantum state has Gaussian-like properties.",
    "Compare this state's Wigner function to that of a thermal state.",
    "Identify whether this is an even or odd cat state.",
    "What is the minimum uncertainty in this state?",
    "Identify whether this state shows quantum interference effects.",
    "Compare this visualization to a displaced squeezed state.",
    "Identify the displacement operator used to generate this state.",
    "How does the Husimi function of this state compare to its Wigner function?",
    "Identify the presence of quantum coherence in this state.",
    "Does this state correspond to a coherent superposition?",
    "Compare this quantum state to a mixed thermal state.",
    "Identify the number of negative regions in this Wigner function.",
    "Is this a highly non-Gaussian quantum state?",
    "Identify the quantum fluctuations in this visualization.",
    "How does this quantum state differ from a vacuum state?",
    "Identify whether this quantum state has a well-defined phase.",
    "Is this Wigner function consistent with a Fock state?",
    "Identify the displacement parameter in this quantum state.",
    "How does this image relate to quantum teleportation states?",
    "Identify whether this state can be used for quantum metrology.",
    "Is this a coherent superposition of multiple Fock states?",
    "Identify whether this quantum state has classical analogs.",
    "Compare this Wigner function to that of a displaced thermal state.",
    "Identify whether this state exhibits quantum squeezing.",
]

# Load CSV file
df = pd.read_csv("metadata-qutip-grayscale.csv")

# Assign random prompts to the "prompt" column
df["prompt"] = [random.choice(qutip_prompts) for _ in range(len(df))]

# Save the updated CSV
df.to_csv("metadata-qutip-grayscale-updated.csv", index=False)

print("Prompts have been randomly assigned and saved.")
