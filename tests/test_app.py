import pytest
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test index route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'TALE Predictive Framework' in response.data

def test_predict_get(client):
    """Test predict route (GET)."""
    response = client.get('/predict')
    assert response.status_code == 200
    assert b'Input Student Data' in response.data