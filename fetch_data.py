from minio import Minio
from minio.error import S3Error
import os
from datetime import datetime

minio_address = 'localhost:8512'
minio_access_key = 'minio'
minio_secret_key = 'minio123'


global client
client = Minio(
        endpoint = minio_address,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

for dataset in ["train", "val"]:
    objects = client.list_objects(bucket_name = "maskdetector", prefix= f"dataset/{dataset}",recursive=True)
    for obj in objects:
        name = obj.object_name
        print("dataset", name)
        client.fget_object("maskdetector", name,"./"+ name)   


def send_data():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    bucket_name = "maskdetector"

    # Make 'mask_detector_dataset' bucket if not exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
    else:
        print("Bucket {} already exists".format(bucket_name))
    parent_path = "/hdd/tuannm82/face-mask-classifier/dataset/test_moana/data_moana"
    for subfol_name in os.listdir(parent_path):
        subfol_path = os.path.join(parent_path, subfol_name)
        for img_name in os.listdir(subfol_path):
            img_path = os.path.join(subfol_path, img_name)
            img_name = "{}_{}".format(datetime.now(), img_name)
            print("img_name: ", img_name)
            client.fput_object(
                bucket_name = "maskdetector",object_name= f"test/{subfol_name}/{img_name}", file_path =img_path,
            )