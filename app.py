import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic dataset with reduced categories
def generate_synthetic_data(n=500):  # Increased data points for more variability
    np.random.seed(42)  # Fixing random seed for reproducibility
    # Creating a synthetic DataFrame with reduced categories for simplicity
    data = pd.DataFrame({
        'Urgency': np.random.randint(1, 6, n),  # Reduced to 5 levels (1-5)
        'Complexity': np.random.randint(1, 6, n),  # Reduced to 5 levels (1-5)
        'Deadline_Days': np.random.randint(1, 16, n),  # Reduced deadline days range (1-15)
        'Task_Length': np.random.randint(1, 51, n),  # Task Length reduced (1-50 hours)
        'Resource_Availability': np.random.randint(1, 6, n),  # Reduced to 5 levels (1-5)
        'Task_Type': np.random.choice(['Bug Fix', 'Feature Development', 'Testing'], n),  # Reduced categories
        'Department': np.random.choice(['IT', 'HR', 'Finance'], n),  # Reduced categories
        'Assigned_To': np.random.choice(['Alice', 'Bob', 'Charlie'], n),  # Reduced categories
        'Priority': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.5, 0.2])  # Reduced categories with probabilities
    })
    return data

data = generate_synthetic_data()  # Generate the synthetic data
st.write("### Sample Synthetic Data")  # Display sample data in Streamlit
st.write(data.head())  # Show first 5 rows of data

# Preprocessing step
def preprocess_data(data):
    encoders = {}  # Dictionary to store label encoders for categorical features
    for col in ['Task_Type', 'Department', 'Assigned_To', 'Priority']:  # Encoding categorical variables
        encoders[col] = LabelEncoder()  # Initialize label encoder
        data[col] = encoders[col].fit_transform(data[col])  # Apply label encoding

    scaler = StandardScaler()  # Initialize StandardScaler to normalize features
    features_to_scale = ['Urgency', 'Complexity', 'Deadline_Days', 'Task_Length', 'Resource_Availability']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])  # Standardize features
    
    return data, encoders, scaler, features_to_scale  # Return the processed data, encoders, scaler, and features to scale

data, encoders, scaler, features_to_scale = preprocess_data(data)  # Apply preprocessing
X = data[['Urgency', 'Complexity', 'Deadline_Days', 'Task_Type', 'Task_Length', 'Resource_Availability', 'Department']]  # Feature matrix
y = data['Priority']  # Target variable (Priority)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 80-20 train-test split

# Initialize models with optimized parameters
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=3)  # RandomForest model
gb_model = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=6, learning_rate=0.03)  # GradientBoosting model
et_model = ExtraTreesClassifier(n_estimators=250, random_state=42, max_depth=12)  # ExtraTrees model

# Combining models into a voting classifier (ensemble method)
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)], voting='soft')
ensemble_model.fit(X_train, y_train)  # Fit ensemble model on training data

# Make predictions on the test data
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model
st.write(f"### Model Accuracy: {accuracy:.2f}")  # Display model accuracy in Streamlit

# Save the trained model, scaler, and encoders for future use
joblib.dump(ensemble_model, "task_prioritizer.pkl")  # Save the trained model
joblib.dump(scaler, "scaler.pkl")  # Save the scaler
joblib.dump(encoders, "encoders.pkl")  # Save the encoders
joblib.dump(features_to_scale, "features_to_scale.pkl")  # Save the list of scaled features

# Streamlit UI for interactive prediction
st.title("Task Prioritizer")  # Set title for the Streamlit app

# User input for the prediction
urgency = st.slider("Urgency (1-5)", 1, 5, 3)  # Urgency slider with 5 levels
complexity = st.slider("Complexity (1-5)", 1, 5, 3)  # Complexity slider with 5 levels
dealine_days = st.number_input("Days Until Deadline (1-15)", 1, 15, 7)  # Input for days until the deadline
task_length = st.number_input("Task Length (in hours, 1-50)", 1, 50, 10)  # Input for task length
resource_availability = st.slider("Resource Availability (1-5)", 1, 5, 3)  # Resource availability slider
task_type = st.selectbox("Task Type", options=["Bug Fix", "Feature Development", "Testing"])  # Dropdown for task type
department = st.selectbox("Department", options=["IT", "HR", "Finance"])  # Dropdown for department
assigned_to = st.selectbox("Assigned To", options=["Alice", "Bob", "Charlie"])  # Dropdown for person assigned

# Encoding the user inputs
task_type_encoded = encoders['Task_Type'].transform([task_type])[0]  # Encode task type
department_encoded = encoders['Department'].transform([department])[0]  # Encode department
assigned_to_encoded = encoders['Assigned_To'].transform([assigned_to])[0]  # Encode assigned person

# Predict when button is clicked
if st.button("Predict Priority"):
    # Prepare input data for prediction
    input_data = np.array([[urgency, complexity, dealine_days, task_type_encoded, task_length, resource_availability, department_encoded]])
    input_df = pd.DataFrame(input_data, columns=['Urgency', 'Complexity', 'Deadline_Days', 'Task_Type', 'Task_Length', 'Resource_Availability', 'Department'])
    input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])  # Apply scaling to input features
    prediction = ensemble_model.predict(input_df)  # Make prediction using the trained model
    predicted_priority = encoders['Priority'].inverse_transform(prediction)[0]  # Decode the predicted priority

    # Display the predicted priority
    st.write(f"### Predicted Priority: {predicted_priority}")
    
   