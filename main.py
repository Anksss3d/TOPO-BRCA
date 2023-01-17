import glob
import cv2
import os

import pandas as pd

from dataset_generation import check_structure, generate_tiles

RAW_DATA_DIR = r"D://Nisha/Dataset_BUSI_with_GT/"
NEW_DATA_DIR = r""
classes = ["benign", "malignant", "normal"]

def remove_masks(DATA_DIR, classes):
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



# generate_dataset2(
#     INPUT_DIR=r"D://Nisha//dataset2/",
#     OUTPUT_DIR=r"D://Nisha/dataset2/",
#     classes=["Benign", "Malignant", "Normal"]
# )
#
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

# generate_tiles(
#     INPUT_DIR=r"/Users/anksss3d/datasets/breast-cesm/CALC/CC/",
#     TILE_DIR=r"/Users/anksss3d/datasets/breast-cesm/CALC/CC_200_features/",
#     CLASS_TO_INDEX={
#         "BENIGN": 0,
#         "MALIGNANT": 1,
#     },
#     INPUT_WIDTH=512,
#     INPUT_HEIGHT=512,
#     TILE_HEIGHT=512,
#     TILE_WIDTH=512,
#     STEP_WIDTH=512,
#     STEP_HEIGHT=512,
#     image_format="jpg",
#     print_frequency=5,
#     thresh=0.99
# )

# generate_tiles(
#     INPUT_DIR=r"/Users/anksss3d/datasets/inbreast2/clean/",
#     TILE_DIR=r"/Users/anksss3d/datasets/inbreast2/tiles_200_features/",
#     CLASS_TO_INDEX={
#         "normal": 0,
#         "mass": 1,
#     },
#     INPUT_WIDTH=224,
#     INPUT_HEIGHT=224,
#     TILE_HEIGHT=224,
#     TILE_WIDTH=224,
#     STEP_WIDTH=224,
#     STEP_HEIGHT=224,
#     image_format="jpg",
#     print_frequency=5,
#     thresh=0.99
# )

# generate_clean_dataset(
#     INPUT_DIR=r"/Users/anksss3d/datasets/inbreast2/dataset/",
#     OUTPUT_DIR=r"/Users/anksss3d/datasets/inbreast2/clean/",
#     classes={
#         "normal": 0,
#         "mass": 1,
#     }
# )

# def generate_cesm_dataset(DATA_DIR, TRAIN_CSV, TEST_CSV, NEW_DATA_DIR, classes):
#     train_df = pd.DataFrame(pd.read_csv(TRAIN_CSV))
#     test_df = pd.DataFrame(pd.read_csv(TRAIN_CSV))
#     df = pd.concat([train_df, test_df], axis=0)
#     # df = df[df["pathology"]==""].replace()
#     df['pathology'] = df['pathology'].replace(['BENIGN_WITHOUT_CALLBACK'], 'BENIGN')
#     print(df.columns)
#     print(len(df))
#     print(df['pathology'])
#     new_df = df[["left or right breast", "image view", "pathology", "image file path"]]
#     n = len(new_df)
#     print(new_df.columns)
#     new_df.columns = [0, 1, 2, 3]
#     k = 0
#     for i in range(n):
#         path = new_df.iloc[i, 3]
#         folder = path.split("/")[-2]
#         images = glob.glob1(DATA_DIR+folder, "*.jpg")
#         img = cv2.imread(DATA_DIR+folder+"/"+images[0])
#         img = cv2.resize(img, (512, 512))
#         cv2.imwrite(NEW_DATA_DIR+new_df.iloc[i, 1]+"/"+new_df.iloc[i, 2]+"/"+str(k)+".jpg", img)
#         k += 1
#         print(k)
#         if len(images)!= 1:
#             print("Problem")
#         # print(len(images))





# generate_cesm_dataset(
#     DATA_DIR=r"/Users/anksss3d/datasets/breast-cesm/jpeg/",
#     TRAIN_CSV=r"/Users/anksss3d/datasets/breast-cesm/csv/mass_case_description_train_set.csv",
#     TEST_CSV=r"/Users/anksss3d/datasets/breast-cesm/csv/mass_case_description_test_set.csv",
#     NEW_DATA_DIR=r"/Users/anksss3d/datasets/breast-cesm/MASS/",
#     classes={
#         "normal": 0,
#         "benign": 1,
#         "malignant": 2,
#     }
# )

