import cv2
from skimage import filters
import glob
import cv2
import os
import pandas as pd
from dataset_generation import check_structure, generate_tiles

def time_break(seconds):
    '''
    Funtion to generate Hrs, Mins, Sec break from seconds
    :param seconds: Input Seconds
    :return: Hrs, Mins and Seconds
    '''
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(((seconds % 3600) % 60))
    return hrs, mins, secs

def tile_content(img):
    '''
    Function to return While tile content values
    :param img: Input tile
    :return: White content Value
    '''
    warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    T = filters.threshold_otsu(warped)
    warped2 = (warped > T).astype("uint8")
    count = warped2.sum()
    perc = round(count / (warped.shape[0] * warped.shape[1]), 2)
    return perc

def remove_masks(DATA_DIR, classes):
    '''
    A function for preprocessing the dataset.
    :param DATA_DIR: Input data directory
    :param classes: Names of the classes
    :return: None
    '''
    xmin, ymin = 10000, 10000
    for clas in classes:
        files = glob.glob1(DATA_DIR+clas, "*.png")
        print(f"No of files: {len(files)}")
        for file in files:
            img = cv2.imread(f"{DATA_DIR}{clas}/{file}")
            # print(img.shape)
            xmin = min(xmin, img.shape[0])
            ymin = min(ymin, img.shape[1])
    print(f"Height min: {xmin}, Width min: {ymin}")

    
def generate_clean_dataset(INPUT_DIR, OUTPUT_DIR, classes):
    '''
    A function for preprocessing the dataset.
    :param DATA_DIR: Input data directory
    :param OUTPUT_DIR: Output data directory
    :param classes: Names of the classes
    :return: None
    '''
    check_structure(INPUT_DIR, OUTPUT_DIR, classes)
    for clas in classes:
        i=0
        files = glob.glob1(INPUT_DIR+clas, "*.png")
        for file in files:
            img = cv2.imread(f"{INPUT_DIR}{clas}/{file}")
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(f"{OUTPUT_DIR}train/{clas}/{i}.jpg", img)
            i+=1

