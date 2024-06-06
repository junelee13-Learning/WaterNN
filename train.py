import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# List of amino acids for the detailed contributions
amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'OTHERS']

def load_data(data_folder):
    total_intrusions = []
    detailed_contributions = []
    labels = []
    
    for subdir, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith('intrusion_summary_') and file.endswith('.txt'):
                with open(os.path.join(subdir, file), 'r') as f:
                    microsphere_intrusions = []
                    microsphere_contributions = {aa: 0.0 for aa in amino_acids}
                    for line in f:
                        if line.startswith('Microsphere'):
                            parts = line.split(':')
                            if len(parts) >= 3 and 'Total Intrusion' in parts[1]:
                                try:
                                    total_intrusion = float(parts[2])
                                    microsphere_intrusions.append(total_intrusion)
                                except ValueError as e:
                                    print(f"Skipping line: {line.strip()} - Error: {e}")
                        elif line.startswith('  Residue'):
                            residue = line.split(':')[1].split(',')[0].strip()
                            residue_total_intrusion = float(line.split(':')[2].strip())
                            if residue not in amino_acids[:-1]:  # Check if residue is not in the list except 'OTHERS'
                                residue = 'OTHERS'
                            microsphere_contributions[residue] += residue_total_intrusion
                    if microsphere_intrusions:
                        total_intrusions.append(microsphere_intrusions[0])  # Only take the first total intrusion value
                        labels.append(1)  # Label for presence of water
                    if microsphere_contributions:
                        detailed_contributions.append([microsphere_contributions[aa] for aa in amino_acids])
    
    X = np.array(total_intrusions).reshape(-1, 1)  # Ensure X is 2D
    X_detailed = np.array(detailed_contributions)
    y = np.array(labels).reshape(-1, 1)  # Labels indicating presence of water
    
    # Generating negative samples (no water)
    num_samples = len(X)
    negative_samples = np.zeros((num_samples, X.shape[1] + X_detailed.shape[1]))
    X_combined = np.hstack((X, X_detailed))
    X_combined = np.vstack((X_combined, negative_samples))
    y_combined = np.vstack((y, np.zeros((num_samples, 1))))
    
    return X_combined, y_combined

def build_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu', name='custom_hidden_layer'))
    model.add(Dense(21, activation='relu', name='amino_acid_layer'))  # 21 neurons for amino acids
    model.add(Dense(32, activation='relu', name='hidden_layer2'))
    model.add(Dense(1, activation='sigmoid', name='output_layer'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def visualize_model(model, output_dir):
    plot_model(model, to_file=os.path.join(output_dir, 'model_architecture.png'), show_shapes=True, show_layer_names=True)
    print(f"Model architecture saved to {os.path.join(output_dir, 'model_architecture.png')}")

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            weight_matrix = weights[0]
            fig, ax = plt.subplots(figsize=(10, 5))
            cax = ax.matshow(weight_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(cax)
            plt.title(f'Weights of {layer.name}')
            plt.xlabel('Output Neurons')
            plt.ylabel('Input Features')
            plt.savefig(os.path.join(output_dir, f'{layer.name}_weights.png'))
            plt.close()
            print(f"Weights of {layer.name} saved to {os.path.join(output_dir, f'{layer.name}_weights.png')}")

def visualize_hidden_layer_weights(model, output_dir, layer_name):
    layer = model.get_layer(name=layer_name)
    weights = layer.get_weights()[0]

    # Debugging: print the shape of the weights matrix
    print(f"Shape of weights in {layer_name}: {weights.shape}")

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.matshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(cax)
    plt.title(f'Weights of {layer_name}')
    plt.xlabel('Neurons')
    plt.ylabel('Features')
    plt.xticks(ticks=np.arange(weights.shape[1]), labels=[f'Neuron {i}' for i in range(weights.shape[1])], rotation=90)

    if layer_name == 'custom_hidden_layer':
        labels = ['Total Intrusion'] + amino_acids  # Including the total intrusion feature
        plt.yticks(ticks=np.arange(weights.shape[0]), labels=labels)
    else:
        plt.yticks(ticks=np.arange(weights.shape[0]), labels=[f'Feature {i}' for i in range(weights.shape[0])])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{layer_name}_weights.png'))
    plt.close()
    print(f"Weights of {layer_name} saved to {os.path.join(output_dir, f'{layer_name}_weights.png')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_folder>")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    trained_modules_dir = 'trained_modules'

    if not os.path.exists(trained_modules_dir):
        os.makedirs(trained_modules_dir)

    X_combined, y_combined = load_data(data_folder)

    input_dim = X_combined.shape[1]
    
    model = build_model(input_dim)

    model.fit(X_combined, y_combined, epochs=10, batch_size=32)

    model.save(os.path.join(trained_modules_dir, 'intrusion_model.keras'))
    visualize_model(model, trained_modules_dir)
    visualize_hidden_layer_weights(model, trained_modules_dir, layer_name='custom_hidden_layer')
    visualize_hidden_layer_weights(model, trained_modules_dir, layer_name='amino_acid_layer')
    visualize_hidden_layer_weights(model, trained_modules_dir, layer_name='hidden_layer2')
