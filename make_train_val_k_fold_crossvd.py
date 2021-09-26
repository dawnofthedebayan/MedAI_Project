import glob
import os 
import cv2 as cv
import numpy as np
from sklearn.model_selection import KFold

dataset_location = "/home/debayan/Desktop/Kvasir-SEG/datasets/dummy/"
datasets = os.listdir(dataset_location)

cross_vd = 5
kf = KFold(n_splits=cross_vd,random_state=123,shuffle=True)
np.random.seed(123)

train_val_split = 0.8
for dataset in datasets: 

    dataset_path = dataset_location + "/" + dataset 

    images = np.array(sorted(glob.glob(dataset_path + "/images/*jpg")))
    masks = np.array(sorted(glob.glob(dataset_path + "/masks/*jpg")))
    train_number = int(len(images) * 0.8)
    val_number_end = int(len(images) * 0.9)
    np.random.shuffle(images)

    train_val_images = np.array(images[:val_number_end])
    test_images = images[val_number_end:]

    cross_val_ds_images_test = dataset_path + "/test/images" 
    cross_val_ds_masks_test = dataset_path + "/test/masks"

    try: 

        os.makedirs(cross_val_ds_images_test)
        os.makedirs(cross_val_ds_masks_test)
            
    except: 

        print("Dir exists")

    for img_loc in test_images: 

        image = cv.imread(img_loc)
        image = cv.resize(image,(512,512))
        image_name = img_loc.split("/")[-1]

        cv.imwrite(cross_val_ds_images_test + "/" + image_name,image)

        mask = cv.imread(img_loc.replace("images","masks"),cv.IMREAD_GRAYSCALE)
        mask = cv.resize(mask,(512,512))

        cv.imwrite(cross_val_ds_masks_test + "/" + image_name,mask)

    
    #for i in range(cross_vd): 

        


    kf.get_n_splits(train_val_images)    
    i = 0
    for train_index, test_index in kf.split(train_val_images):
        
        X_train, X_val = train_val_images[train_index], train_val_images[test_index]
        print(X_train[:5],X_val[:5])
        print("######")
        cross_val_ds_images_train = dataset_path + "/{}/images_train".format(i) 
        cross_val_ds_masks_train = dataset_path + "/{}/masks_train".format(i) 
        cross_val_ds_images_val = dataset_path + "/{}/images_val".format(i) 
        cross_val_ds_masks_val = dataset_path + "/{}/masks_val".format(i) 

        try: 

            os.makedirs(cross_val_ds_images_train)
            os.makedirs(cross_val_ds_masks_train)
            os.makedirs(cross_val_ds_images_val)
            os.makedirs(cross_val_ds_masks_val)

        except: 

            print("Dir exists")

        i += 1
        for img_loc in X_train: 

            image = cv.imread(img_loc)
            image = cv.resize(image,(512,512))
            image_name = img_loc.split("/")[-1]

            cv.imwrite(cross_val_ds_images_train + "/" + image_name,image)

            mask = cv.imread(img_loc.replace("images","masks"),cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask,(512,512))

            cv.imwrite(cross_val_ds_masks_train + "/" + image_name,mask)


        for img_loc in X_val: 

            image = cv.imread(img_loc)
            image = cv.resize(image,(512,512))
            image_name = img_loc.split("/")[-1]

            cv.imwrite(cross_val_ds_images_val + "/" + image_name,image)

            mask = cv.imread(img_loc.replace("images","masks"),cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask,(512,512))

            cv.imwrite(cross_val_ds_masks_val + "/" + image_name,mask)
        
     

    
