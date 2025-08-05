import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

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

# Create a pipeline for text processing and model training
# TF-IDF works on how rare words are, which is exactly what we want for ticket classification
# Pipeline helps avoid data leakage and ensures same transformations for train/test data
pipeline = Pipeline([
    # TF-IDF Vectorizer - converts text to numerical features
    ('tfidf', TfidfVectorizer(
        max_features=10000,      # Limit vocabulary size to reduce noise
        ngram_range=(1, 2),      # Use unigrams and bigrams for better context
        stop_words='english',    # Remove common English words
        max_df=0.95,            # Ignore terms in more than 95% of documents
        min_df=2                # Ignore terms in less than 2 documents
    )),
    # Logistic Regression classifier
    ('classifier', LogisticRegression(
        max_iter=1000,          # Iterations for convergence
        C=1.0                   # Regularization parameter to prevent overfitting
    ))
])

# Train the model
# Pipeline automatically applies TF-IDF transformation then trains the classifier
pipeline.fit(X_train, y_train)

# Make predictions on test data
# Pipeline automatically applies same TF-IDF transformation to test data
y_pred = pipeline.predict(X_test)

# Convert predictions back to original category names (optional for viewing)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Calculate and display accuracy
# Accuracy = correct predictions / total predictions
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and label encoder for future use
# Pipeline contains both TF-IDF vectorizer and trained classifier
# Label encoder is needed to convert predictions back to category names
joblib.dump(pipeline, 'models/model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
print("Model and label encoder saved successfully!")