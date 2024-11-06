import tensorflow as tf

class RollingAverageModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='loss', save_best_only=True, mode='min', verbose=1, rolling_epochs=10):
        super(RollingAverageModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.rolling_epochs = rolling_epochs
        self.losses = []

        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")

        self.best = float('inf') if mode == 'min' else -float('inf')
        self.compare = lambda x, y: x < y if mode == 'min' else x > y

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)

        if current_loss is None:
            return

        # Add current loss to the list and maintain the rolling window
        self.losses.append(current_loss)
        if len(self.losses) > self.rolling_epochs:
            self.losses.pop(0)

        # Calculate rolling average
        rolling_average = sum(self.losses) / len(self.losses)

        if self.compare(current_loss, rolling_average):
            if self.verbose > 0:
                print(
                    f"\nEpoch {epoch + 1}: {self.monitor} improved from {rolling_average:.4f} to {current_loss:.4f}, saving model to {self.filepath}")
            self.model.save(self.filepath)
            self.best = current_loss
        elif self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {rolling_average:.4f}")
