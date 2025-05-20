from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

class DLModel:
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the model."""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return self

    def predict(self, X):
        """Make predictions."""
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def save(self, path):
        """Save the model."""
        self.model.save(path)

    def load(self, path):
        """Load the model."""
        self.model = load_model(path)
        return self