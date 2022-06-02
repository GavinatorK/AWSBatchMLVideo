# AWSBatchMLVideo
AWS Batch for video inference at scale

## Code Structure 

**Creates two docker images, all the files for each image are in corresponding folder**

### fsxPredictDocker
This folder has all the files needed to create docker image for splitting video into frames, load up a simple tflite model and perform inference and save the frames back with bounding boxes.

**Note:

### frames2vid

folder contains all the files to create a docker image to compiles frames back into video after inference
