import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from keras import callbacks


class weights_visualization_callback(callbacks.Callback):
    def __init__(self, num_of_layers):
        self.weights_history = [[] for _ in range(num_of_layers)]
        self.weights_index = np.arange(0, num_of_layers * 2, 2)
        self.num_of_layers = num_of_layers

    def on_epoch_end(self, batch, logs):
        weights_biases = self.model.get_weights()
        # print('on_epoch_end() model.weights:', weights_1)

        for i in range(self.num_of_layers):
            cur_weight_i = self.weights_index[i]
            self.weights_history[i].append(weights_biases[cur_weight_i])

    def get_weights(self):
        return self.weights_history


class weights_history_visualizer:
    def __init__(self, all_weights_history, mode='3d'):
        self.all_weights_history = all_weights_history
        self.num_of_layers = len(all_weights_history)
        self.mode = mode
        self.animations = []
        pass

    def update_2d(self, frame, ax):
        for j in range(self.num_of_layers):
            ax[j].cla()
            weights_history = self.all_weights_history[j]
            ax[j].imshow(weights_history[frame],
                         cmap='hot')
            ax[j].set_title(f"hidden layer {j + 1}")
            ax[j].set_xlabel("hidden layer size")
            ax[j].set_ylabel("input dimension")

        ax[0].annotate(f"Epoch = {(frame + 1):d}", xy=(0.1, 0.1),
                       xycoords='figure fraction')
        return ax

    def update_3d(self, frame, ax):
        for j in range(self.num_of_layers - 1):
            ax[j].cla()
            weights_history = self.all_weights_history[j]
            input_dim, hidden_dim = weights_history[0].shape
            plot_X = np.arange(hidden_dim)
            plot_Y = np.arange(input_dim)
            plot_X, plot_Y = np.meshgrid(plot_X, plot_Y)
            ax[j].plot_surface(plot_X, plot_Y, weights_history[frame],
                               cmap='hot')
            ax[j].set_title(f"hidden layer {j + 1}")
            ax[j].set_xlabel("hidden layer size")
            ax[j].set_ylabel("input dimension")

        ax[0].annotate(f"Epoch = {(frame + 1):d}", xy=(0.1, 0.1),
                       xycoords='figure fraction')
        ax[-1].cla()
        ax[-1].imshow(self.all_weights_history[-1][frame], cmap='hot')
        ax[-1].set_title(f"hidden layer {self.num_of_layers}")
        ax[-1].set_xlabel("hidden layer size")
        ax[-1].set_ylabel("input dimension")
        return ax

    def visualize(self, interval=1, frametime=16):
        if self.mode == '3d':
            self.visualize_3d(interval, frametime)
        elif self.mode == '2d':
            self.visualize_2d(interval, frametime)

    def visualize_2d(self, interval, frametime):
        fig, ax = plt.subplots(1, self.num_of_layers,
                               figsize=(6 * self.num_of_layers, 8))
        fig.suptitle("weights in hidden layer over epochs")
        epochs = len(self.all_weights_history[0])
        # initialize plots
        for j in range(self.num_of_layers):
            ax[j].remove()
            ax[j] = fig.add_subplot(1, self.num_of_layers, j + 1)
            weights_history = self.all_weights_history[j]
            input_dim, hidden_dim = weights_history[0].shape
            plot_X = np.arange(hidden_dim)
            plot_Y = np.arange(input_dim)
            plot_X, plot_Y = np.meshgrid(plot_X, plot_Y)
            v_min = np.array(weights_history).min()
            v_max = np.array(weights_history).max()
            print(f"minimum weight: {v_min:.2f}")
            print(f"maximum weight: {v_max:.2f}")
            weights_grid = ax[j].imshow(weights_history[0], cmap='hot')
            fig.colorbar(weights_grid, ax=ax[j], shrink=0.5)

        ani = FuncAnimation(fig=fig, func=self.update_2d, fargs=(ax, ),
                            frames=range(0, epochs, interval),
                            interval=frametime)
        self.animations = ani

        plt.show()

        pass

    def visualize_3d(self, interval, frametime):
        fig, ax = plt.subplots(1, self.num_of_layers,
                               figsize=(6 * self.num_of_layers, 8))
        fig.suptitle("weights in hidden layer over epochs")
        epochs = len(self.all_weights_history[0])
        for j in range(self.num_of_layers - 1):
            ax[j].remove()
            ax[j] = fig.add_subplot(1, self.num_of_layers, j + 1,
                                    projection='3d')
            weights_history = self.all_weights_history[j]
            input_dim, hidden_dim = weights_history[0].shape
            plot_X = np.arange(hidden_dim)
            plot_Y = np.arange(input_dim)
            plot_X, plot_Y = np.meshgrid(plot_X, plot_Y)
            v_min = np.array(weights_history).min()
            v_max = np.array(weights_history).max()
            print(f"minimum weight: {v_min:.2f}")
            print(f"maximum weight: {v_max:.2f}")
            weights_surf = ax[j].plot_surface(plot_X, plot_Y,
                                              weights_history[0],
                                              cmap='hot',
                                              vmin=v_min, vmax=v_max)
            fig.colorbar(weights_surf, ax=ax[j], shrink=0.5)

        last_layer_w = self.all_weights_history[-1]
        ax[-1].remove()
        ax[-1] = fig.add_subplot(1, self.num_of_layers, self.num_of_layers)
        v_min = np.array(last_layer_w).min()
        v_max = np.array(last_layer_w).max()
        print(f"minimum weight: {v_min:.2f}")
        print(f"maximum weight: {v_max:.2f}")
        weights_grid = ax[-1].imshow(last_layer_w[0], cmap='hot')
        fig.colorbar(weights_grid, ax=ax[-1], shrink=0.5)

        ani = FuncAnimation(fig=fig, func=self.update_3d, fargs=(ax, ),
                            frames=range(0, epochs, interval),
                            interval=frametime)
        self.animations = ani

        plt.show()

        pass

    def save(self, filename):
        self.animations.save(filename, fps=15)
        pass
