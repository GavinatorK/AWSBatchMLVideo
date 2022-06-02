import json
import boto3
from urllib.parse import unquote_plus


def lambda_handler(event, context):
    # logic to get the filename from s3
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])

    print(bucket, key)
    batch = boto3.client('batch', 'us-east-1')
    #s3 path to move the file to sync with FSx
    s3DestinationPath="input/incoming/"+key.split("/")[-1]
    #relative path inside container
    containerPath="/home/data/input/incoming/"+key.split("/")[-1]
    framePath="/home/data/input/frames/"+key.split("/")[-1]
    
    
    s3 = boto3.resource('s3')
    copy_source = {
        'Bucket': bucket,
        'Key': key
    }
    
    copyResp=s3.meta.client.copy(copy_source, "batch-video-ml", s3DestinationPath)
    print(copyResp)
    
    Inference_job = batch.submit_job(
        jobName='do-inference',
        jobQueue='fsx-test-queue',
        jobDefinition='fsxObject:1',
        containerOverrides={
            'environment': [
                {
                    'name': 'vidpath',
                    'value': containerPath
                },
                {   'name': 'mlType',
                    'value':'detect'
                    
                },
                {
                    'name':'model',
                    'value':'SSD'
                }]
        })
    
    print(Inference_job)
    
    
    vidCompile_job=batch.submit_job(
        jobName='compile-frames',
        jobQueue='fsx-test-queue',
        dependsOn=[
            {
                 'jobId': Inference_job['jobId']
            },
         ],
        jobDefinition='fsx-complie-vid:1',
        containerOverrides={
            'environment': [
                {
                    'name': 'vidpath',
                    'value': framePath
                }]
        })
        
    print(vidCompile_job)
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": ""
    }


