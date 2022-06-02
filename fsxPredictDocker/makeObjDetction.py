import os
import pathlib
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def convertToTFLite(saved_model_dir):
    
    import tensorflow as tf

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    return None

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)
'''
MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
'''

# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)
'''
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)
'''

'''
from object_detection.utils import label_map_util
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile
'''


def loadModelandLabels(model='SSD'):
    with open("/home/data/input/tfModel/labelMap.json", 'r') as f:
        category_index=json.load(f)



    import time
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    if model=='SSD':
        PATH_TO_SAVED_MODEL = "/home/data/input/tfModelSSD/saved_model"
    else:
        PATH_TO_SAVED_MODEL = "/home/data/input/tfModel/saved_model"
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn, category_index

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

detect_fn, category_index=loadModelandLabels(model='SSD')
def runObjInference(frame,savePath, model='SSD'):

    print('Running inference for {}... '.format(model), end='')
    
    #image_np = load_image_into_numpy_array(image_path)
    image_np=frame
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    print("done with predictions for {}".format(savePath))
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    new_detections = {}
    new_detections['boxes'] = []
    new_detections['scores'] = []
    new_detections['classes'] = []

    for index, score in enumerate(detections['detection_scores']):
        if score > 0.5:
            new_detections['boxes'].append(detections['detection_boxes'][index])
            new_detections['scores'].append(detections['detection_scores'][index])
            new_detections['classes'].append(detections['detection_classes'][index])

    new_detections['boxes'] = np.array(new_detections['boxes'])
    new_detections['scores'] = np.array(new_detections['scores'])
    new_detections['classes'] = np.array(new_detections['classes'])    


    image_np_with_detections = image_np.copy()
    print("about to save the image with boxes")
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          new_detections['boxes'],
          new_detections['classes'],
          new_detections['scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    try:
        plt.savefig(savePath)
    except Exception as e:
        print(e)
        saveDir="/".join(savePath.split("/")[:-1])
        pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath)
    print('Done')

