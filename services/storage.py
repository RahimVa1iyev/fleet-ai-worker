import os
import boto3
from botocore.client import Config

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def download_frame(r2_key: str) -> bytes:
    """R2-dən frame-i yüklə və bytes olaraq qaytar"""
    client = get_s3_client()
    bucket = os.environ["R2_BUCKET"]
    response = client.get_object(Bucket=bucket, Key=r2_key)
    return response["Body"].read()