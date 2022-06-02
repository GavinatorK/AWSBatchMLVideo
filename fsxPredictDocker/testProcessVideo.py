import cv2  # still used to save images out
import os
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys
import tensorflow as tf
from makePrediction import *
#Add Inference code here so it can be called after the file is saved.

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
#Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def prepImgForServingNumpy(img):
    img = img/255
    img = np.expand_dims(img, axis=0)
    input_data = np.float32(img)
    return input_data    

    

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):

    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, width=224, height=224, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    if every > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
                saved_count += 1  # increment our counter by one

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture

            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                save_path = os.path.join(frames_dir, video_filename,
                                         "{:010d}.jpg".format(index))  # create the save path
                
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    #cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
                    saved_count += 1  # increment our counter by one
                    # let's dp inference on framw we just saved
                    interpreter.set_tensor(input_details[0]['index'], prepImgforServingNumpy(frame.asnumpy))
                    interpreter.invoke()
                    if any(interpreter.get_tensor(output_details[0]['index'])[0])>0.9:
                        print("found something")
                        cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
                    else:
                        print("found nothing")
                        cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2GRAY )) 
                        # and return the count of the images we saved

    return saved_count


def print_progress(iteration,total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param comp: remaining chunks
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """
    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    if total==0 and iteration==0:
        percents=format_str.format(100)
        filled_length=100
    else:
        percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
        filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout

def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):

    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total
    print("frame chunks "+str(len(frame_chunks))) 
    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print(i,f)
            print_progress(i,len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames


if __name__ == '__main__':

    # test it
    x=video_to_frames(video_path='/home/data/testmp4.mp4', frames_dir='/home/data/test_frames', overwrite=True, every=5, chunk_size=1000)
    print(x)

