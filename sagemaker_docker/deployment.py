from sagemaker.model import Model
import json
import boto3

def get_secret():
    client = boto3.client('secretsmanager', region_name='us-east-2')
    response = client.get_secret_value(
        SecretId="arn:aws:secretsmanager:us-east-2:339713173631:secret:OPENAI_API_KEY-B2TLX4"
    )
    return json.loads(response['SecretString'])

secret = get_secret()
openai_api_key = secret.get('OPENAI_API_KEY')

region = 'us-east-2'
image_uri = '339713173631.dkr.ecr.us-east-2.amazonaws.com/duke_fashivly_ml_repo:latest'
role = 'arn:aws:iam::339713173631:role/service-role/AmazonSageMaker-ExecutionRole-20250112T100139'

sm_model = Model(
    image_uri=image_uri,
    role=role,
    name='sam-segmentation',
    env={'OPENAI_API_KEY': openai_api_key}
)

predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='sam-segmentation'
)
