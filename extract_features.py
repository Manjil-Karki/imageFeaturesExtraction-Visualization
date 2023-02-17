import pandas as pd
import numpy as np
import cv2
import os
import h5py
from skimage.filters.rank import entropy
from skimage.morphology import disk


def getImagesPath():
    file_names = os.listdir(r"visualize/assets/images")
    index = [f.split(".")[0] for f in file_names]
    file_names = pd.Series(file_names)
    images_relative_path = "./visualize/assets/images/" + file_names
    return images_relative_path, index


def initMetaDataDF():
    df = pd.DataFrame() 
    paths, index = getImagesPath()
    df["relative_path"] = paths
    df["index"] = index
    df.set_index("index", inplace=True)
    
    # Read all images and store as array of numpy arrays
    print("Reading All Images...\n")
    images = [cv2.imread(path) for path in paths]
    
    images = np.array(images, dtype = object)
    # Get dimensions of all images
    print("Reading Dimensions of Images...\n")
    dimensions = np.array(list(map(lambda x: x.shape, images)))

    # Update Dataframe
    df[["Height", "Width", "Channel"]] = dimensions

    # Add orentaiton column
    df["Orientation"] = np.where(df["Height"] > df["Width"], "Potrait", "landscape")
    return images, df


def storeimage(image, img_path):
    f_name = img_path.split('/')[-1].split(".")[0]
    f_path = r"./visualize/features/"+f_name+".h5"
    with h5py.File(f_path, "w") as f:
        image[:, :, [0, -1]] = image[:, :, [-1, 0]]
        f.create_dataset("Image", data=image, dtype=np.uint8)
    return True

def getColorSpace(image, img_path):
    # reshape image 
    flat_img = image.reshape(-1, image.shape[-1])
    # Obtain A color only once
    unique_colors = np.unique(flat_img, axis = 0)

    f_name = img_path.split('/')[-1].split(".")[0]
    f_path = r"./visualize/features/"+f_name+".h5"
    with h5py.File(f_path, "a") as f:
        # had to do this due to vectorization creating multiple instances
        if "ColorSpace" in f:
            del f["ColorSpace"]
        unique_colors[:, [0, -1]] = unique_colors[:, [-1, 0]]
        f.create_dataset("ColorSpace", data=unique_colors)
    return True

def getColorHist(image, img_path):
    red = cv2.calcHist([image], [2], None, [256], [0, 256]).reshape(256)
    green = cv2.calcHist([image], [1], None, [256], [0, 256]).reshape(256)
    blue = cv2.calcHist([image], [0], None, [256], [0, 256]).reshape(256)
    hist_arr = np.array([red, green, blue]).T
    f_name = img_path.split('/')[-1].split(".")[0]
    f_path = r"./visualize/features/"+f_name+".h5"
    with h5py.File(f_path, "a") as f:        
        if "ColorHist" in f:
            del f["ColorHist"]
        hist_arr[:, [0, -1]] = hist_arr[:, [-1, 0]]
        f.create_dataset("ColorHist", data=hist_arr)
    return True

def getEdges(image, img_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,150,200)
    f_name = img_path.split('/')[-1].split(".")[0]
    f_path = r"./visualize/features/"+f_name+".h5"
    with h5py.File(f_path, "a") as f:        
        if "Edges" in f:
            del f["Edges"]
        f.create_dataset("Edges", data=edges)
    return True

def getEntropy(image, img_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_entropy = entropy(gray, disk(1))
    f_name = img_path.split('/')[-1].split(".")[0]
    f_path = r"./visualize/features/"+f_name+".h5"
    with h5py.File(f_path, "a") as f:        
        if "Entropy" in f:
            del f["Entropy"]
        f.create_dataset("Entropy", data=img_entropy)
    return True

def getKeypoints(image, img_path):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect keypoints using SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw the keypoints on the image
    keypoints = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints = cv2.cvtColor(keypoints, cv2.COLOR_BGR2RGB)

    f_name = img_path.split('/')[-1].split(".")[0]
    f_path = r"./visualize/features/"+f_name+".h5"
    with h5py.File(f_path, "a") as f:        
        if "Keypoints" in f:
            del f["Keypoints"]
        f.create_dataset("Keypoints", data=keypoints)
    return f_path

def collectStoreFeatures(images, df):
    if not os.path.exists(r"./visualize/features"):
        os.mkdir(r"./visualize/features")

    print("Storing images...\n")
    store = np.vectorize(storeimage)
    store(images, df["relative_path"])

    # Obtaining Color Space
    # All colors in image are stored in the form of n X 3 array of pixel colors
    print("Obtaining Image colour space...\n")
    ColorSpace = np.vectorize(getColorSpace)
    ColorSpace(images, df["relative_path"])


    # Obtaining data for image color histogram
    print("Obtaining Data for color Histogram...\n")
    ColorHistogram = np.vectorize(getColorHist)
    ColorHistogram(images, df["relative_path"])

    # Obtaining Edges
    print("Extracting Edges...\n")
    edges = np.vectorize(getEdges)
    edges(images, df["relative_path"])

    # Obtaining Edges
    print("Extracting Entropy...\n")
    entropy = np.vectorize(getEntropy)
    entropy(images, df["relative_path"])

    # Obtaining Edges
    print("Extracting Keypoints...\n")
    keypoints = np.vectorize(getKeypoints)
    f_paths = keypoints(images, df["relative_path"])


    return f_paths



if __name__ == "__main__":

    images, df = initMetaDataDF()

    # images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

    f_paths = collectStoreFeatures(images, df)


    df["Features_Fpath"] = f_paths
    df.to_csv("./visualize/dataframe.csv")
    print(df.head())