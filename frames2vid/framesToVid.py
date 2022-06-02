import re
import cv2
# import numpy as np
import os
from os.path import isfile, join
import argparse

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]


def compileFrames(pathIn,pathOut,fps=6):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    sorted_images = sorted(files, key=natural_sort_key)
    # for sorting the file names properly
    # files.sort(key=lambda x: x[5:-4])
    pathOut=pathOut+pathIn.split("/")[-1]
    for i in range(len(sorted_images)):
        filename = pathIn+"/" + sorted_images[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__=="__main__":
    # test it
    my_parser = argparse.ArgumentParser(description='provide path to video file')
    my_parser.add_argument('--vid', metavar='vid', type=str, help='path to video frame to process')

    args=my_parser.parse_args()
    print("Args Passed "+ str(args))
    x=compileFrames(pathIn=args.vid, pathOut='/home/data/input/infVids/',fps=6)
    print(x)

