import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_merge_data(data_dir='data/raw'):
    """Load and merge OULAD datasets."""
    student_info = pd.read_csv(f'{data_dir}/studentInfo.csv')
    vle = pd.read_csv(f'{data_dir}/vle.csv')
    assessments = pd.read_csv(f'{data_dir}/assessments.csv')
    student_assessment = pd.read_csv(f'{data_dir}/studentAssessment.csv')
    
    # Aggregate VLE clicks
    vle_agg = vle.groupby('id_student')['sum_click'].sum().reset_index()
    vle_agg.rename(columns={'sum_click': 'total_clicks'}, inplace=True)
    
    # Compute average assessment score
    assessment_scores = student_assessment.merge(assessments, on='id_assessment')
    assessment_scores['weighted_score'] = assessment_scores['score'] * assessment_scores['weight'] / 100
    avg_scores = assessment_scores.groupby('id_student')['weighted_score'].mean().reset_index()
    avg_scores.rename(columns={'weighted_score': 'avg_assessment_score'}, inplace=True)
    
    # Merge datasets
    data = student_info.merge(vle_agg, on='id_student', how='left').merge(avg_scores, on='id_student', how='left')
    return data

def preprocess_data(data):
    """Preprocess data for modeling."""
    # Handle missing values
    data['total_clicks'].fillna(0, inplace=True)
    data['avg_assessment_score'].fillna(data['avg_assessment_score'].mean(), inplace=True)
    
    # Binarize target variable
    data['final_result'] = data['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
    
    # Encode categorical variables
    categorical_cols = ['gender', 'region', 'highest_education', 'age_band']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Scale numerical features
    numerical_cols = ['total_clicks', 'avg_assessment_score', 'studied_credits', 'num_of_prev_attempts']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data