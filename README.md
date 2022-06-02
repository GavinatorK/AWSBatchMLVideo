# AWSBatchMLVideo
AWS Batch for video inference at scale

## Code Structure 

**Creates two docker images, all the files for each image are in corresponding folder**

### fsxPredictDocker
This folder has all the files needed to create docker image for splitting video into frames, load up a simple tflite model and perform inference and save the frames back with bounding boxes.

**Note: In this example, we placed the tflite model directly into the imag, this is not practical for more complex and bigger models, It is advisable to store it in FSX for Lustre or Amazon S3 and load it in from there**


### frames2vid

folder contains all the files to create a docker image to compiles frames back into video after inference
