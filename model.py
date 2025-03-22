import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
heart_data = pd.read_csv('data.csv')

# Splitting features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Function for making predictions
def predict_heart_disease(data):
    model = pickle.load(open('model.pkl', 'rb'))
    input_data = np.asarray(data).reshape(1, -1)
    prediction = model.predict(input_data)
    return "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have a Heart Disease"
