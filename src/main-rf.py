import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow

import dagshub
dagshub.init(repo_owner='Pratt33', repo_name='smart-ticket-routing-mlflow', mlflow=True)

# Set MLflow tracking URI to local mlruns directory
# tracking server allows to store artifacts, parameters, and metrics
#mlflow.set_tracking_uri("file:./mlruns")#local storage for MLflow
#mlflow.set_tracking_uri("http://localhost:5000")# local server for MLflow
#mlflow.set_tracking_uri("https://dagshub.com/Pratt33/smart-ticket-routing-mlflow.mlflow")# dagshub server for MLflow
mlflow.set_tracking_uri("http://ec2-3-108-54-126.ap-south-1.compute.amazonaws.com:5000/")# remote server for MLflow with aws

# Load the ticket dataset
df = pd.read_csv('data/raw/all_tickets_processed_improved_v3.csv')

# Encode labels - convert text categories to numbers
# LabelEncoder converts categorical labels to numerical values (e.g., 'Hardware' -> 0, 'Access' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Topic_group'])

# Split data into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures reproducible results
# stratify=y_encoded maintains the same proportion of each class in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['Document'], 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
max_features =8000  # Limit vocabulary size to reduce noise
max_iter = 1000  # Maximum iterations for convergence in Logistic Regression
C = 1.0  # Regularization parameter to prevent overfitting

# Create a pipeline for text processing and model training
# TF-IDF works on how rare words are, which is exactly what we want for ticket classification
# Pipeline helps avoid data leakage and ensures same transformations for train/test data
pipeline = Pipeline([
    # TF-IDF Vectorizer - converts text to numerical features
    ('tfidf', TfidfVectorizer(
        max_features=max_features,      # Limit vocabulary size to reduce noise
        ngram_range=(1, 2),      # Use unigrams and bigrams for better context
        stop_words='english',    # Remove common English words
        max_df=0.95,            # Ignore terms in more than 95% of documents
        min_df=2                # Ignore terms in less than 2 documents
    )),
    # Random Forest classifier
    ('classifier', RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=10,          # Limit depth to prevent overfitting
        min_samples_split=10  # Minimum samples to split an internal node
    ))
])

mlflow.set_experiment("ticket-rf")

# Start a single MLflow run for everything
with mlflow.start_run():
    # Enable autolog but disable dataset logging to avoid Series warnings
    mlflow.autolog(log_datasets=False)
    
    # Train the model - no more dataset warnings
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Log custom artifacts (confusion matrix, code, tags)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix.png')
    
    mlflow.log_artifact('visualizations/confusion_matrix.png')
    mlflow.log_artifact(__file__)
    
    # Log datasets manually (as you're already doing)
    train_df = pd.DataFrame({'Document': X_train, 'Topic_group': label_encoder.inverse_transform(y_train)})
    test_df = pd.DataFrame({'Document': X_test, 'Topic_group': label_encoder.inverse_transform(y_test)})
    
    train_dataset = mlflow.data.from_pandas(train_df)
    test_dataset = mlflow.data.from_pandas(test_df)
    
    mlflow.log_input(train_dataset, "train")
    mlflow.log_input(test_dataset, "test")
    
    mlflow.set_tag('author', 'Pratt33')
    mlflow.set_tag('model_type', 'Random Forest')
    
    print("Experiment logged to MLflow successfully")

# Convert predictions back to original category names
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Save the trained model and label encoder for future use
joblib.dump(pipeline, 'models/model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("Model and label encoder saved successfully!")