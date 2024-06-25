import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def load_processed_data(data_folder):
    data = []
    labels = []
    
    for subdir, _, files in os.walk(data_folder):
        for file in files:
            if file == 'coordinates_1.csv' or file == 'coordinates_0.csv':
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                
                features = df.values.flatten()
                if len(features) == 10 * (6 + 21 + 3 + 3):  # Ensure correct number of features
                    data.append(features)
                    if file == 'coordinates_1.csv':
                        labels.append(1)  # Label for presence of water
                    else:
                        labels.append(0)  # Label for absence of water

    X = np.array(data)
    y = np.array(labels).reshape(-1, 1)  # Labels indicating presence of water
    
    return X, y

def build_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu', name='hidden_layer'))
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_folder>")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    trained_modules_dir = 'trained_modules'

    if not os.path.exists(trained_modules_dir):
        os.makedirs(trained_modules_dir)

    X, y = load_processed_data(data_folder)

    input_dim = X.shape[1]
    
    model = build_model(input_dim)

    model.fit(X, y, epochs=10, batch_size=32)

    model.save(os.path.join(trained_modules_dir, 'water_model.keras'))
    visualize_model(model, trained_modules_dir)
