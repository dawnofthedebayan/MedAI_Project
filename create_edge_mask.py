import glob
import cv2 as cv
from utils.utils import create_dir
import numpy as np
from tqdm import tqdm


for validation_set in range(5):

    

    mask = "/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/train/mask/*jpg".format(validation_set)
    save_path = "/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/train/edge_mask/{}".format(validation_set,"")

    create_dir(save_path)
    save_path = "/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/train/edge_mask/{}"
    mask_paths = glob.glob(mask)
    for mask_pt in tqdm(mask_paths,total=len(mask_paths)):

        name = mask_pt.split("/")[-1]
        img = cv.imread(mask_pt,cv.IMREAD_GRAYSCALE)

        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        edge_mask = np.zeros((img.shape[0],img.shape[1],3))
        
        for c in contours:
        
            cv.drawContours(edge_mask,[c], 0, (255,255,255),3)

        edge_mask = cv.cvtColor(edge_mask.astype('uint8'),cv.COLOR_RGB2GRAY)
        cv.imwrite(save_path.format(validation_set,name),edge_mask)

    #break


