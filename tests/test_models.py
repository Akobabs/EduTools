import pytest
import numpy as np
from src.models.ml_model import MLModel

def test_ml_model():
    """Test ML model training and prediction."""
    X_train = np.random.rand(10, 8)
    y_train = np.random.randint(0, 2, 10)
    X_test = np.random.rand(5, 8)
    
    model = MLModel(model_type='rf')
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    
    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)