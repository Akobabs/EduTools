import pytest
import pandas as pd
from src.data_preprocessing import preprocess_data

def test_preprocessing():
    """Test data preprocessing."""
    data = pd.DataFrame({
        'gender': ['M', 'F'],
        'region': ['North', 'South'],
        'highest_education': ['A Level', 'HE Qualification'],
        'age_band': ['0-35', '35-55'],
        'num_of_prev_attempts': [0, 1],
        'studied_credits': [60, 120],
        'total_clicks': [100, None],
        'avg_assessment_score': [80, None],
        'final_result': ['Pass', 'Fail']
    })
    
    processed_data = preprocess_data(data.copy())
    
    assert processed_data['total_clicks'].isnull().sum() == 0
    assert processed_data['avg_assessment_score'].isnull().sum() == 0
    assert processed_data['final_result'].isin([0, 1]).all()
    assert processed_data['gender'].dtype == int