import random
import csv
import os
import json
from collections import defaultdict
from qiskit.circuit.random import random_circuit
from qiskit.qasm2 import dumps
from qiskit import transpile
from qiskit_aer import Aer
import math
import time
import matplotlib.pyplot as plt
import requests

from quantastica.qps_api import QPS

QPS.save_account("rxBbSgeGgFPJiHZk9DAQJwn5hiqGmoiuzb")

# Configuration
MIN_QUBITS = 1            # Minimum number of qubits to generate    
MAX_QUBITS = 20                 # Maximum number of qubits to generate
BASE_SAMPLES = 3               # Base samples for max qubits (will propagate down)
MIN_DEPTH = 1                   # Minimum circuit depth
MAX_DEPTH = 15                  # Maximum circuit depth
TOTAL_SAMPLES = 100000          # Total number of samples to generate
IMAGE_DIR = 'Dataset/Source1/images/{qubit_number}-qubit-generated'    # Directory to store circuit images
CSV_FILENAME = 'Dataset/Source1/quantum_circuits_{qubit_number}_qubit.csv'  # Output CSV file

def calculate_distribution(num_qubits=20, total_data=100000):
    # Calculate the geometric progression ratio
    ratio = 4/3
    
    # Calculate the sum of the geometric series coefficients
    coefficients = [ratio**i for i in range(num_qubits)]
    total_coeff = sum(coefficients)
    
    # Calculate base value for first qubit
    base = total_data / total_coeff
    
    # Calculate exact values (before rounding)
    exact_values = [base * ratio**i for i in range(num_qubits)]
    
    # Round values and track the difference
    rounded = [math.ceil(v) for v in exact_values]
    current_total = sum(rounded)
    
    # Adjust to reach exactly 100000
    while current_total > total_data:
        max_index = rounded.index(max(rounded))
        rounded[max_index] -= 1
        current_total -= 1
        
    while current_total < total_data:
        min_index = rounded.index(min(rounded))
        rounded[min_index] += 1
        current_total += 1
    
    return rounded

def generate_circuit_data():
    """Main function to generate circuits and CSV data"""
    sample_distribution = calculate_distribution(MAX_QUBITS, TOTAL_SAMPLES)
    counter = defaultdict(int)
    
    execution_counter = 24300
    qubit_executed = 20

    # Generate circuits for each qubit count
    for qubits in range(MIN_QUBITS, MAX_QUBITS+1):
        if qubits < qubit_executed:
            num_samples = sample_distribution[qubits-1]
            continue
        
        csv_filename = CSV_FILENAME.format(qubit_number=qubits)
        
        # Check if the file exists and is not empty
        file_exists = os.path.isfile(csv_filename) and os.path.getsize(csv_filename) > 0

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            if not file_exists:
                writer.writerow(['image_path1', 'image_path2', 'number_of_qubits', 'openqasm', 'qiskit', 'quil', 'cirq', 'qobj', 'number_of_shots', 'ground_truth'])
            
            # Create output directory if it doesn't exist
            image_dir = IMAGE_DIR.format(qubit_number=qubits)
            os.makedirs(image_dir, exist_ok=True)
            num_samples = sample_distribution[qubits-1]
            
            for _ in range(num_samples):
                # Generate random depth for each circuit
                depth = random.randint(MIN_DEPTH, MAX_DEPTH)
                
                # Create unique identifier for this qubit/depth combination
                key = (qubits, depth)
                serial_num = counter[key] + 9999
                counter[key] += 1
                
                # Generate random circuit
                circ = random_circuit(qubits, depth, measure=True)
                
                # Create filename with sorting-friendly format
                mpl_filename = f"q{qubits:02}_d{depth:03}_s{serial_num:04}_mpl.png"
                mpl_image_path = os.path.join(image_dir, mpl_filename)
                
                # Save circuit image
                fig = circ.draw(output='mpl')
                fig.savefig(mpl_image_path)
                plt.close(fig)
                
                # Create filename with sorting-friendly format
                latex_filename = f"q{qubits:02}_d{depth:03}_s{serial_num:04}_latex.png"
                latex_image_path = os.path.join(image_dir, latex_filename)
                
                # Save circuit image
                circ.draw(output='latex', filename=latex_image_path)
                
                # Convert to OpenQASM
                qasm_str = dumps(circ)

                try:
                    qiskit_str = QPS.converter.convert(qasm_str, "qasm", "qiskit")
                    
                    quil_str = QPS.converter.convert(qasm_str, "qasm", "quil")
                    
                    cirq_str = QPS.converter.convert(qasm_str, "qasm", "cirq")  
                    
                    qobj_str = QPS.converter.convert(qasm_str, "qasm", "qobj")
                
                    # Simulate circuit
                    simulator = Aer.get_backend('aer_simulator')
                    compiled = transpile(circ, simulator)
                    result = simulator.run(compiled).result()
                    counts = result.get_counts()
                    
                    # Calculate probabilities
                    total_shots = sum(counts.values())
                    probabilities = {state: count/total_shots for state, count in counts.items()}
                    
                    # Write to CSV
                    writer.writerow([
                        mpl_image_path,
                        latex_image_path,
                        qubits,
                        qasm_str,
                        qiskit_str,
                        quil_str,
                        cirq_str,
                        qobj_str,
                        total_shots,
                        json.dumps(probabilities, indent=4)
                    ])
                except requests.exceptions.ReadTimeout as err:
                    print(f"Error: The request to {err.request.url} timed out after 3 seconds.")
                except requests.exceptions.ConnectionError:
                    print(f"Error: Failed to connect to {err.request.url}. Check your internet connection.")
                except requests.exceptions.HTTPError as err:
                    print(f"HTTP Error {err.response.status_code}: {err.response.reason}")
                except requests.exceptions.RequestException as err:
                    print(f"An unexpected error occurred: {err}")
                
                execution_counter += 1
                
                # if execution_counter over in every 100 steps, do sleep 3 seconds
                if execution_counter % 100 == 0:
                    print(f"Generated {execution_counter} circuits for {qubits}-qubit circuits from total {num_samples} samples")
                    time.sleep(1)
                    

if __name__ == '__main__':
    # result = calculate_distribution(MAX_QUBITS, TOTAL_SAMPLES)
    generate_circuit_data()
    print(f"Generated dataset with {BASE_SAMPLES} base samples in {CSV_FILENAME}")