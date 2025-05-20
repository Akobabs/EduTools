from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

class MLModel:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            self.model = XGBClassifier(random_state=42)
        else:
            raise ValueError("Unsupported model type. Use 'rf' or 'xgb'.")

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def save(self, path):
        """Save the model."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load the model."""
        self.model = joblib.load(path)
        return self