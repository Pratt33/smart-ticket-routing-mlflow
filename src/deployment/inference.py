import mlflow

mlflow.set_tracking_uri("http://ec2-3-108-54-126.ap-south-1.compute.amazonaws.com:5000/")

# Load the registered model
model = mlflow.sklearn.load_model("models:/tickets-rf-v2-model/latest")

# Test prediction
sample = ["Server down, users cannot login"]
prediction = model.predict(sample)
print(f"Prediction: {prediction}")