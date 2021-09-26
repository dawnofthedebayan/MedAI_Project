
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import glob

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    RandomCrop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    size = (512, 512)
    crop_size = (448, 448)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        image_name = x.split("/")[-1].split(".")[0]
        mask_name = y.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if x.shape[0] >= size[0] and x.shape[1] >= size[1]:
            if augment == True:
                ## Crop
                x_min = 0
                y_min = 0
                x_max = x_min + size[0]
                y_max = y_min + size[1]

                aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                augmented = aug(image=x, mask=y)
                x1 = augmented['image']
                y1 = augmented['mask']

                # Random Rotate 90 degree
                aug = RandomRotate90(p=1)
                augmented = aug(image=x, mask=y)
                x2 = augmented['image']
                y2 = augmented['mask']

                ## ElasticTransform
                aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                augmented = aug(image=x, mask=y)
                x3 = augmented['image']
                y3 = augmented['mask']

                ## Grid Distortion
                aug = GridDistortion(p=1)
                augmented = aug(image=x, mask=y)
                x4 = augmented['image']
                y4 = augmented['mask']

                ## Optical Distortion
                aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
                augmented = aug(image=x, mask=y)
                x5 = augmented['image']
                y5 = augmented['mask']

                ## Vertical Flip
                aug = VerticalFlip(p=1)
                augmented = aug(image=x, mask=y)
                x6 = augmented['image']
                y6 = augmented['mask']

                ## Horizontal Flip
                aug = HorizontalFlip(p=1)
                augmented = aug(image=x, mask=y)
                x7 = augmented['image']
                y7 = augmented['mask']

                ## Grayscale
                #x8 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                #y8 = y

                ## Grayscale Vertical Flip
                #aug = VerticalFlip(p=1)
                #augmented = aug(image=x8, mask=y8)
                #x9 = augmented['image']
                #y9 = augmented['mask']

                ## Grayscale Horizontal Flip
                #aug = HorizontalFlip(p=1)
                #augmented = aug(image=x8, mask=y8)
                #x10 = augmented['image']
                #y10 = augmented['mask']

                aug = RandomBrightnessContrast(p=1)
                augmented = aug(image=x, mask=y)
                x11 = augmented['image']
                y11 = augmented['mask']
                #
                aug = RandomGamma(p=1)
                augmented = aug(image=x, mask=y)
                x12 = augmented['image']
                y12 = augmented['mask']
                #
                aug = HueSaturationValue(p=1)
                augmented = aug(image=x, mask=y)
                x13 = augmented['image']
                y13 = augmented['mask']

                aug = RGBShift(p=1)
                augmented = aug(image=x, mask=y)
                x14 = augmented['image']
                y14 = augmented['mask']

                aug = RandomBrightness(p=1)
                augmented = aug(image=x, mask=y)
                x15 = augmented['image']
                y15 = augmented['mask']
                #
                aug = RandomContrast(p=1)
                augmented = aug(image=x, mask=y)
                x16 = augmented['image']
                y16 = augmented['mask']

                #aug = ChannelShuffle(p=1)
                #augmented = aug(image=x, mask=y)
                #x17 = augmented['image']
                #y17 = augmented['mask']

                aug = CoarseDropout(p=1, max_holes=10, max_height=32, max_width=32)
                augmented = aug(image=x, mask=y)
                x18 = augmented['image']
                y18 = augmented['mask']

                aug = GaussNoise(p=1)
                augmented = aug(image=x, mask=y)
                x19 = augmented['image']
                y19 = augmented['mask']

                aug = MotionBlur(p=1, blur_limit=7)
                augmented = aug(image=x, mask=y)
                x20 = augmented['image']
                y20 = augmented['mask']
                #
                #aug = MedianBlur(p=1, blur_limit=11)
                #augmented = aug(image=x, mask=y)
                #x21 = augmented['image']
                #y21 = augmented['mask']
                #
                aug = GaussianBlur(p=1, blur_limit=11)
                augmented = aug(image=x, mask=y)
                x22 = augmented['image']
                y22 = augmented['mask']

                ##
                aug = CenterCrop(256, 256, p=1)
                augmented = aug(image=x, mask=y)
                x23 = augmented['image']
                y23 = augmented['mask']

                aug = CenterCrop(384, 384, p=1)
                augmented = aug(image=x, mask=y)
                x24 = augmented['image']
                y24 = augmented['mask']

                aug = CenterCrop(448, 448, p=1)
                augmented = aug(image=x, mask=y)
                x25 = augmented['image']
                y25 = augmented['mask']

                ## x23 Vertical Flip
                aug = VerticalFlip(p=1)
                augmented = aug(image=x23, mask=y23)
                x26 = augmented['image']
                y26 = augmented['mask']

                ## x23 Horizontal Flip
                aug = HorizontalFlip(p=1)
                augmented = aug(image=x23, mask=y23)
                x27 = augmented['image']
                y27 = augmented['mask']

                ## x24 Vertical Flip
                aug = VerticalFlip(p=1)
                augmented = aug(image=x24, mask=y24)
                x28 = augmented['image']
                y28 = augmented['mask']

                ## x24 Horizontal Flip
                aug = HorizontalFlip(p=1)
                augmented = aug(image=x24, mask=y24)
                x29 = augmented['image']
                y29 = augmented['mask']

                ## x25 Vertical Flip
                aug = VerticalFlip(p=1)
                augmented = aug(image=x25, mask=y25)
                x30 = augmented['image']
                y30 = augmented['mask']

                ## x25 Horizontal Flip
                aug = HorizontalFlip(p=1)
                augmented = aug(image=x25, mask=y25)
                x31 = augmented['image']
                y31 = augmented['mask']

                images = [
                    x, x1, x2, x3, x4, 
                    x5,
                    x6, x7, 
                    #x8, x9,
                    #x10,
                    x11,# 
                    x12, x13,
                    x14,
                    x15, x16,
                    #x17,
                    x18, 
                    x19,
                    x20, #x21, 
                    x22,
                    x23, x24, x25, x26, x27, x28, x29,
                    x30, x31
                ]
                masks  = [
                    y, y1, y2, y3, y4, y5, 
                    y6, y7, 
                    #y8, y9,
                    #y10,
                    y11, #
                    y12, y13,
                    y14,
                    y15, y16,
                    #y17, 
                    y18, 
                    y19,
                    y20, #y21, 
                    y22,
                    y23, y24, y25, y26, y27, y28, y29,
                    y30, y31
                ]

            else:
                images = [x]
                masks  = [y]

            idx = 0
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            if len(images) == 1:
                tmp_image_name = f"{image_name}.jpg"
                tmp_mask_name  = f"{mask_name}.jpg"
            else:
                tmp_image_name = f"{image_name}_{idx}.jpg"
                tmp_mask_name  = f"{mask_name}_{idx}.jpg"
            
            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)
            
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def main():
    #np.random.seed(42)
    for validation_set in range(5):
        # path = "../../Medico/Kvasir-SEG/"
        
        #train_images_path = "/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/CVC-ClinicDB/{}/images_train/*png".format(validation_set)
        #train_masks_path  = "/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/CVC-ClinicDB/{}/masks_train/*png".format(validation_set)

        #val_images_path = "/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/CVC-ClinicDB/{}/images_val/*png".format(validation_set)
        #val_masks_path  = "/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/CVC-ClinicDB/{}/masks_val/*png".format(validation_set)

        #t_images = np.array(sorted(glob.glob(train_images_path)))
        #t_masks = np.array(sorted(glob.glob(train_masks_path)))
        #v_images = np.array(sorted(glob.glob(val_images_path)))
        #v_masks = np.array(sorted(glob.glob(val_masks_path)))
        seed = np.random.randint(2147483647) 
        np.random.seed(seed)

        images = np.array(sorted(glob.glob("/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/kvasir-instruments/images/*jpg")))
        masks = np.array(sorted(glob.glob("/home/debayan/Desktop/Kvasir-SEG/datasets/final_datasets/kvasir-instruments/masks/*png")))

        #np.random.seed(123)
        
        rand_idxs = np.random.permutation(len(images))
        
        train_idxs = rand_idxs[:int(len(images)*0.9)] # choosing the first 8000 as training set
        #val_idxs = rand_idxs[int(len(images)*0.8):int(len(images)*0.9)] # choosing last 2000 as validation set
        test_idxs = rand_idxs[int(len(images)*0.9):] # choosing last 2000 as validation set
        
        train_images = images[train_idxs]
        train_masks = masks[train_idxs]

        #val_images = images[val_idxs]
        #val_masks = masks[val_idxs]
        
        test_images = images[test_idxs]
        test_masks = masks[test_idxs]

        print("Training Images:",len(train_images))
        print("Validation Images:",len(test_images))
        #print("Test Images:",len(test_idxs))

        #create_dir("new_data/CVC-ClinicDB/{}/train/image/".format(validation_set))
        #create_dir("new_data/CVC-ClinicDB/{}/train/mask/".format(validation_set))
        #create_dir("new_data/CVC-ClinicDB/{}/validate/image/".format(validation_set))
        #create_dir("new_data/CVC-ClinicDB/{}/validate/mask/".format(validation_set))
        create_dir("/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/train/image/".format(validation_set))
        create_dir("/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/train/mask/".format(validation_set))
        create_dir("/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/validate/image/".format(validation_set))
        create_dir("/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/validate/mask/".format(validation_set))
        #augment_data(t_images, t_masks, "new_data/CVC-ClinicDB/{}/train/".format(validation_set), augment=True)
        #augment_data(v_images, v_masks, "new_data/CVC-ClinicDB/{}/validate/".format(validation_set), augment=False)
        #augment_data(test_images, test_masks, "new_data/KVASIR_SEG/test/{}/", augment=False)

        augment_data(train_images, train_masks, "/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/train".format(validation_set), augment=True)
        augment_data(test_images, test_masks, "/home/debayan/Desktop/MedAI_Project/new_data/nora_instruments_new/{}/validate/".format(validation_set), augment=False)

if __name__ == "__main__":
    main()