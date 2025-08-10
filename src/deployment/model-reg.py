import mlflow

mlflow.set_tracking_uri("http://ec2-3-108-54-126.ap-south-1.compute.amazonaws.com:5000/")

run_id = '17eeebdb7b9b4ebda86b3ece6d5c0261'
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri, "tickets-rf-final")
print(f"Registered: {result.name} v{result.version}")