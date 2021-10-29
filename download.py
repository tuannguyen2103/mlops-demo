import boto3
from mlflow.tracking import MlflowClient
import mlflow.pytorch

client = MlflowClient("http://10.61.185.121:5120")
model_path = client.get_registered_model('MLops-demo2').latest_versions[0].source

model_url = model_path.split("s3://mlflow/")[-1]

s3 = boto3.client('s3', endpoint_url='http://10.61.185.121:8512', aws_access_key_id='minio', aws_secret_access_key='minio123')
# print("Downloading the latest model...")
s3.download_file('mlflow', model_url + "/data/model.pth", 'model/data/model.pth')
s3.download_file('mlflow', model_url + "/data/pickle_module_info.txt", 'model/data/pickle_module_info.txt')
s3.download_file('mlflow', model_url + "/MLmodel", 'model/MLmodel')

