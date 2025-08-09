import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow

import dagshub
dagshub.init(repo_owner='Pratt33', repo_name='smart-ticket-routing-mlflow', mlflow=True)
#mlflow.set_tracking_uri("https://dagshub.com/Pratt33/smart-ticket-routing-mlflow.mlflow")# dagshub server for MLflow

mlflow.set_tracking_uri("http://ec2-3-108-54-126.ap-south-1.compute.amazonaws.com:5000/")# remote server for MLflow with aws

# Load the ticket dataset
df = pd.read_csv('data/raw/all_tickets_processed_improved_v3.csv')

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Topic_group'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Document'], 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.95,
        min_df=2
    )),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for faster execution
param_grid = {
    # TF-IDF parameters
    'tfidf__max_features': [5000, 8000],

    # Random Forest parameters
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5, 10],      
    'classifier__min_samples_leaf': [1, 2]
}

# Initialize GridSearchCV with CV folds
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=2,                    # CV set to 2-fold
    scoring='accuracy',
    n_jobs=-1,              
    verbose=2,              
    return_train_score=True 
)

mlflow.set_experiment('ticket-rf-hp-fast')

with mlflow.start_run():
    # Perform GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get results
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    #log params
    mlflow.log_params(best_params)

    #log metrics
    mlflow.log_metrics({
        'best_cv_score': float(best_cv_score),
        'test_accuracy': float(test_accuracy)
    })

    #log data
    train_df = pd.DataFrame({'Document': X_train, 'Topic_group': label_encoder.inverse_transform(y_train)})
    test_df = pd.DataFrame({'Document': X_test, 'Topic_group': label_encoder.inverse_transform(y_test)})
    
    train_dataset = mlflow.data.from_pandas(train_df)
    test_dataset = mlflow.data.from_pandas(test_df)
    
    mlflow.log_input(train_dataset, "train")
    mlflow.log_input(test_dataset, "test")

    #log source code
    mlflow.log_artifact(__file__)

    #log model
    mlflow.sklearn.log_model(best_model, "model")

    #set tag
    mlflow.set_tags({"author": "Pratt33"})

    print("GridSearchCV completed and logged to MLflow!")

# Save best model and results
joblib.dump(best_model, 'models/best_rf_gridsearch_fast.pkl')
joblib.dump(label_encoder, 'models/label_encoder_gridsearch_fast.pkl')