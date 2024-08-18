import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from tensorflow.keras import callbacks

weights_history_1 = []
weights_history_2 = []

class weights_visualization_callback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights_1, biases_1 = self.model.layers[1].get_weights()
        weights_2, biases_2 = self.model.layers[2].get_weights()
        weights_history_1.append(weights_1.copy())
        weights_history_2.append(weights_2.copy())

def load_processed_data(data_folder):
    data = []
    labels = []
    
    for subdir, _, files in os.walk(data_folder):
        for file in files:
            if file == 'coordinates_1.csv' or file == 'coordinates_0.csv':
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                features = df.values.flatten()
                if len(features) == 10 * (13 + 28 + 3 + 3):  # Ensure correct number of features
                    data.append(features)
                    if file == 'coordinates_1.csv':
                        labels.append(1)  # Label for presence of water
                    else:
                        labels.append(0)  # Label for absence of water

    X = np.array(data)
    y = np.array(labels).reshape(-1, 1)  # Labels indicating presence of water
    
    return X, y

def build_model(input_dim, hidden_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_dim, activation='relu', name='hidden_layer_1'))
    model.add(Dense(256, activation='relu', name='hidden_layer_2'))
    model.add(Dense(128, activation='relu', name='hidden_layer_3'))
    model.add(Dense(64, activation='relu', name='hidden_layer_4')) # Additional hidden layer
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
    hidden_dim = 512 #128
    epochs = 80

    model = build_model(input_dim, hidden_dim)

    weights_cb = weights_visualization_callback()
    history = model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[weights_cb])

    model.save(os.path.join(trained_modules_dir, 'water_model.keras'))
    visualize_model(model, trained_modules_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history.history['loss'], label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    plt.show()

    # Visualizing weights for the first hidden layer
    # fig_a1, ax_a1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 12))
    # plot_X1 = np.arange(hidden_dim)
    # plot_Y1 = np.arange(input_dim)
    # plot_X1, plot_Y1 = np.meshgrid(plot_X1, plot_Y1)  # Correct order for weights shape
    weights_array_1 = np.array(weights_history_1)

    v_min1 = weights_array_1.min()
    v_max1 = weights_array_1.max()
    print(f"minimum weight (layer 1): {v_min1:.2f}")
    print(f"maximum weight (layer 1): {v_max1:.2f}")

    # weights_surf_1 = ax_a1.plot_surface(plot_X1, plot_Y1, weights_array_1[0], cmap='hot', vmin=v_min1, vmax=v_max1)
    # colorbar_1 = fig_a1.colorbar(weights_surf_1, ax=ax_a1, shrink=0.5)
    # ax_a1.set_title("Weights in hidden layer 1 over epochs")
    # ax_a1.set_xlabel("Hidden layer size")
    # ax_a1.set_ylabel("Input dimension")
    # artists_1 = []
    # for i in range(epochs):
    #     ax_a1.clear()
    #     weights_surf_1 = ax_a1.plot_surface(plot_X1, plot_Y1, weights_array_1[i], cmap='hot')
    #     epoch_index_1 = ax_a1.annotate(f"Epoch = {(i + 1):d}", xy=(0.1, 0.1), xycoords='figure fraction')
    #     artists_1.append([weights_surf_1, epoch_index_1])

    # ani_1 = ArtistAnimation(fig=fig_a1, artists=artists_1, interval=60)
    # plt.show()

    # Visualizing weights for the second hidden layer
    # fig_a2, ax_a2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 12))
    # plot_X2 = np.arange(48)
    # plot_Y2 = np.arange(hidden_dim)
    # plot_X2, plot_Y2 = np.meshgrid(plot_X2, plot_Y2)  # Correct order for weights shape
    weights_array_2 = np.array(weights_history_2)

    v_min2 = weights_array_2.min()
    v_max2 = weights_array_2.max()
    print(f"minimum weight (layer 2): {v_min2:.2f}")
    print(f"maximum weight (layer 2): {v_max2:.2f}")

    # weights_surf_2 = ax_a2.plot_surface(plot_X2, plot_Y2, weights_array_2[0], cmap='hot', vmin=v_min2, vmax=v_max2)
    # colorbar_2 = fig_a2.colorbar(weights_surf_2, ax=ax_a2, shrink=0.5)
    # ax_a2.set_title("Weights in hidden layer 2 over epochs")
    # ax_a2.set_xlabel("Hidden layer 2 size")
    # ax_a2.set_ylabel("Hidden layer 1 size")
    # artists_2 = []
    # for i in range(epochs):
    #     ax_a2.clear()
    #     weights_surf_2 = ax_a2.plot_surface(plot_X2, plot_Y2, weights_array_2[i], cmap='hot')
    #     epoch_index_2 = ax_a2.annotate(f"Epoch = {(i + 1):d}", xy=(0.1, 0.1), xycoords='figure fraction')
    #     artists_2.append([weights_surf_2, epoch_index_2])

    # ani_2 = ArtistAnimation(fig=fig_a2, artists=artists_2, interval=60)
    # plt.show()
