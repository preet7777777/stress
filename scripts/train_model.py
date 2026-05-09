import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_and_save_mock_model():
    # Generate synthetic data based on research paper features
    # Features: EDA_mean, EDA_std, HR_mean, HRV_RMSSD, Temp_mean, Acc_SMA
    np.random.seed(42)
    n_samples = 1000
    
    # Baseline data (Stress = 0)
    baseline_data = np.random.normal(loc=[0.5, 0.05, 70, 50, 32, 0.1], 
                                     scale=[0.1, 0.01, 5, 10, 0.5, 0.05], 
                                     size=(n_samples // 2, 6))
    
    # Stress data (Stress = 1) - Higher EDA, Higher HR, Lower HRV, Lower Temp
    stress_data = np.random.normal(loc=[2.5, 0.8, 95, 25, 30, 0.15], 
                                   scale=[0.5, 0.2, 10, 5, 0.5, 0.05], 
                                   size=(n_samples // 2, 6))
    
    X = np.vstack((baseline_data, stress_data))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    # Save
    if not os.path.exists('stress-ml/models'):
        os.makedirs('stress-ml/models')
        
    joblib.dump(model, 'stress-ml/models/stress_model.joblib')
    joblib.dump(scaler, 'stress-ml/models/scaler.joblib')
    print("✅ Mock model and scaler saved successfully in stress-ml/models/")

if __name__ == "__main__":
    train_and_save_mock_model()
