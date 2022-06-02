import tensorflow as tf
import requests
import json
import numpy as np
import argparse
import os
import shutil
import boto3

def prepImgforServing(img_path, IMG_SIZE=224, CHANNELS=3):
    
    # Read and prepare image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    input_data = np.float32(img)
    return input_data


s3_client = boto3.client('s3')

def download_dir(prefix, local, bucket, client=s3_client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)

    

def runInference():
    #download_dir("input/incoming", "/home/data", "batch-video-ml")
    img_path="/home/data/input/incoming"
    for each in os.listdir(img_path):
        print("going to make inference call")
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors() 
        #Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()   
    
        interpreter.set_tensor(input_details[0]['index'], prepImgforServing(img_path+"/"+each))
        interpreter.invoke()
        if any(interpreter.get_tensor(output_details[0]['index'])[0])>0.9:
            shutil.move(img_path+"/"+each, "/home/data/input/processed/"+each)
            #s3_client.upload_file(img_path+"/"+each, "batch-video-ml", "processsed/"+each)
        
    return interpreter.get_tensor(output_details[0]['index'])

def runObjDetection():
    pass


'''
if __name__=='__main__':
    #my_parser = argparse.ArgumentParser(description='Load an image and get inference for jaic')
    #my_parser.add_argument('--img', metavar='img', type=str, help='path to video frame to process')

    #args=my_parser.parse_args()
    #print("Args Passed "+ str(args))
    preds=runInference()
    print({"conex":preds[0][0], "person":preds[0][1]})

'''
