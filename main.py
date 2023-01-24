import glob
import cv2
import os

import pandas as pd

from dataset_generation import check_structure, generate_tiles

RAW_DATA_DIR = r"D://Nisha/Dataset_BUSI_with_GT/"
NEW_DATA_DIR = r""
classes = ["benign", "malignant", "normal"]

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


def generate_dataset2(INPUT_DIR, OUTPUT_DIR, classes):
    df = pd.DataFrame(pd.read_csv(INPUT_DIR+"annotations.csv"))
    n, m = df.shape
    p = 0
    for i in range(n):
        img_name = f"{INPUT_DIR}Images/{df.iloc[i, 2]}/{df.iloc[i, 0].replace(' ', '')}.jpg"
        # print(img_name)
        img = cv2.imread(img_name)
        # print(img.shape)
        img = cv2.resize(img, (400, 1400))
        cv2.imwrite(f"{INPUT_DIR}/{df.iloc[i, 3]}/{df.iloc[i, 2]}/{df.iloc[i, 4]}/{p}.jpg", img)
        p+=1



generate_tiles(
    INPUT_DIR=r"/Users/anksss3d/datasets/breast-cesm/CALC/MLO/",
    TILE_DIR=r"/Users/anksss3d/datasets/breast-cesm/CALC/MLO_200_features/",
    CLASS_TO_INDEX={
        "BENIGN": 0,
        "MALIGNANT": 1,
    },
    INPUT_WIDTH=512,
    INPUT_HEIGHT=512,
    TILE_HEIGHT=512,
    TILE_WIDTH=512,
    STEP_WIDTH=512,
    STEP_HEIGHT=512,
    image_format="jpg",
    print_frequency=5,
    thresh=0.99
)

