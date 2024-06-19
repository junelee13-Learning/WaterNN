import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
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
                
                features = df.values.reshape(10, 40)  # Reshape the dataframe to 10x40 (10 atoms, 40 features each)
                if features.shape == (10, 40):  # Ensure correct shape
                    data.append(features)
                    if file == 'coordinates_1.csv':
                        labels.append(1)  # Label for presence of water
                    else:
                        labels.append(0)  # Label for absence of water

    X = np.array(data)
    y = np.array(labels).reshape(-1, 1)  # Labels indicating presence of water
    print(y)
    
    return X, y

def build_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    flattened_input = Flatten()(input_layer)
    
    # Encoder
    encoded = Dense(256, activation='relu')(flattened_input)
    encoded = Dense(128, activation='relu')(encoded)
    
    # Latent space
    latent_space = Dense(64, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(128, activation='relu')(latent_space)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)
    decoded_output = Reshape(input_shape)(decoded)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded_output)
    
    # Encoder model
    encoder = Model(input_layer, latent_space)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def build_classifier(encoder, input_shape):
    for layer in encoder.layers:
        layer.trainable = False
    
    encoded_input = Input(shape=input_shape)
    x = encoder(encoded_input)
    
    # Classifier
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    classifier = Model(encoded_input, output)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

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

    input_shape = (X.shape[1], X.shape[2])  # Input shape is (10, 40)
    
    autoencoder, encoder = build_autoencoder(input_shape)
    autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

    classifier = build_classifier(encoder, input_shape)
    classifier.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    autoencoder.save(os.path.join(trained_modules_dir, 'autoencoder_model.keras'))
    classifier.save(os.path.join(trained_modules_dir, 'classifier_model.keras'))
    visualize_model(autoencoder, trained_modules_dir)
    visualize_model(classifier, trained_modules_dir)
