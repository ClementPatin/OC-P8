# imports
import tensorflow as tf
import keras
import segmentation_models as sm

import albumentations as A

import os

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



class generator(keras.utils.Sequence) :
    '''
    data generator for cityscapes dataset :
        - gets every images (and masks) paths
        - splits them using scikit learn "train_test_split" if necessary
        - reads images per batch
        - applies data augmentation if needed
        - applies image preprocessing
        - regroups classes into macro categories using a given dictionnary {macro_class_name : list of classes}
        - put categories in their own channels

        
    parameters :
    ------------
    batch_size - int
    images_path - string : path to the root folder containing images
    masks_path - string : path to the root folder containing masks
    image_size - tuple : (heigh, width)
    n_images - int or None : number of images to keep. By default : None (no sampling)
    cats - dict or None : {macro_class_name : list of classes}. By default : None (no classes modifications)
    augmentation - Albumentations transform or None : For image augmentation. by default : None
    backbone - string or None : Name of the model architecture (backbone) that the data generator will used to train. Used for image preocessing. By default : None
    shuffle - bool : To decide wether or not to shuffle the images between each epoch. By default : False
    split - bool : wether or not to split the images/masks into two sets. By default : False
    split_test_size - float : for scikit learn "train_test_split". By default : 0.1
    split_keep - string : "split_train" or "split_test". To decide which part the generator will use. By default : "split_trian"
    split_rs - int : for scikit learn "train_test_split" parammeter "random_state". Important for keeping the same spliting between two generators. By default : 16

    '''

    def __init__(self, batch_size, images_path, masks_path, image_size, which_set, n_images=None, cats=None, augmentation=None, backbone=None, shuffle=False, split=False, split_test_size = 0.1, split_keep="split_train", split_rs=16) :
        '''
        creates lists of paths (for images and for masks) and handles sampling and splitting
        '''
        self.batch_size = batch_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.which_set = which_set
        self.n_images = n_images
        self.cats = cats
        self.augmentation = augmentation
        self.backbone = backbone
        self.shuffle = shuffle
        self.split = split
        self.split_keep = split_keep
        self.split_rs = split_rs

        # initiate lists to store paths, 1 for images, 1 for masks
        self.images_path_list = []
        self.masks_path_list = []

        # for each folder and each list
        for path, l in zip(
            [self.images_path, self.masks_path],
            [self.images_path_list, self.masks_path_list]
        ) :
            # get set folder
            set_path = os.path.join(path,which_set)
            # put each city folder path in a list
            cities_list = os.listdir(set_path)
            cities_path_list = [os.path.join(set_path,city) for city in cities_list]
            # add images paths contained in each city folder
            # (carefull with mask, take PNG with "labelIds" in its name)
            l.extend([
                    os.path.join(city_path,img_file_name) \
                    for city_path in cities_path_list \
                        for img_file_name in os.listdir(city_path) \
                            if ("leftImg8bit" in img_file_name) or ("labelIds" in img_file_name)
            ])
            # sort
            l.sort()

        # handle n_images
        if n_images :
            self.images_path_list = self.images_path_list[:n_images]
            self.masks_path_list = self.masks_path_list[:n_images]
        
        # handle split if neccessary
        if split :
            im_train, im_test = train_test_split(self.images_path_list, test_size=split_test_size, random_state=split_rs)
            mk_train, mk_test = train_test_split(self.masks_path_list, test_size=split_test_size, random_state=split_rs)
            if split_keep == "split_train" :
                self.images_path_list = im_train
                self.masks_path_list = mk_train
            else :
                self.images_path_list = im_test
                self.masks_path_list = mk_test

        # indexes in an attribute
        self.indexes = np.arange(len(self.images_path_list))
        # apply "on_epoch_end" for shuffling (see below)
        self.on_epoch_end()

    def __len__(self) :
        '''
        Denotes the number of batches per epoch
        '''
        # return int(np.floor(len(self.images_path_list)/self.batch_size))
        return len(self.images_path_list) // self.batch_size
    
    def __getitem__(self, batch_idx) :
        '''
        Creates a batch : loading, augmentation, image processing, labels changing, putting labels in seprated channels and returning batch images and masks
        '''
        # indexes for this batch
        start = batch_idx * self.batch_size
        stop = (batch_idx +1) * self.batch_size
        indexes_for_this_batch = self.indexes[start:stop]
        # initiate lists for this batch
        batch_images = []
        batch_masks = []
        # build batch
        for i in indexes_for_this_batch :
            # load image and mask as arrays 
            image = keras.preprocessing.image.load_img(self.images_path_list[i], target_size=self.image_size)
            image = keras.preprocessing.image.img_to_array(image, dtype="uint8")
            mask = keras.preprocessing.image.load_img(self.masks_path_list[i], color_mode = "grayscale", target_size=self.image_size)
            mask = keras.preprocessing.image.img_to_array(mask)

            # apply augmentation
            if self.augmentation :
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # apply preprocessing, for backbone compatibility
            if self.backbone :
                preprocessor = sm.get_preprocessing(self.backbone)
                preprocessor = A.Lambda(image=preprocessor)
                preprocessor = A.Compose([preprocessor])
                sample = preprocessor(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # simplify categories using self.cats values
            if self.cats :
                for k, list_of_labels in enumerate(self.cats.values()) :
                    mask = np.where(
                        np.isin(
                            mask, 
                            list_of_labels
                            ), 
                        k, 
                        mask
                        ).astype("uint8")

            # cast mask to categorical
            mask = keras.utils.to_categorical(mask, num_classes=len(self.cats), dtype="float32")

            # add to lists
            batch_images.append(image)
            batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)
    
    def on_epoch_end(self):
        '''Callback function to shuffle indexes at each epoch'''
        if self.shuffle :
            self.indexes = np.random.permutation(self.indexes)





def baseline_model(input_shape, num_classes):
    '''
    create a simple segmentation model based on convolutionnal layers

    parameters :
    ------------
    input_shape - tuple : for Input of the Sequential model. (height, width, channels)
    num_classes - int : Number of classes. For the last Convolutionnal layer (channel wize pooling)

    returns :
    ---------
    Sequential model
    '''


    layers = [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
        keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.UpSampling2D(size=(2,2)),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.UpSampling2D(size=(2,2)),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
        keras.layers.Conv2D(filters=num_classes, kernel_size=3, strides=1, padding="same", activation="softmax"),       
    ]



    return keras.models.Sequential(layers, name="baseline_model")





def testAugmentations(gen, trans_dict, idx, cats_colors) :
    '''
    test augmentation using Albumentations and plot the result for an image and its mask

    parameters :
    ------------
    gen - generator custom class instance
    trans_dict - dict of Albumentations transforms, {"name" : transform, ...}
    idx - int : index in generator "gen"
    cats_colors - array of shape (num of classes, 3). RGB colors for each grayscaled class in masks

    display :
    ---------
    matplotlib figure : with (number of tranforms x 4) axes :
        - image
        - mask
        - augmented image
        - augmented mask
    '''

    # imports 
    import matplotlib.pyplot as plt
    import numpy as np


    # get image and its mask from "gen"
    image = gen[0][0][idx].astype("uint8")
    mask = gen[0][1][idx].astype("uint8")

    # create figure
    fig, axs = plt.subplots(len(trans_dict),4,figsize=(20,10/4*len(trans_dict)))

    # iterate on each transformation in trans_list
    for i,(trans_name, trans) in enumerate(trans_dict.items()) :
        # create a compose object  from "trans" and use it on image and mask
        aug = A.Compose(trans)
        sample = aug(image=image, mask=mask)
        image_aug = sample["image"]
        # for mask, get each pixel label using "argmax" and map with cats_colors
        mask_col = cats_colors[np.argmax(mask, axis=2)]
        mask_aug = cats_colors[np.argmax(sample["mask"], axis=2)]

        # imshow and titles
        for j,(im,title) in enumerate(zip([image, mask_col, image_aug, mask_aug],["image", "mask", "augmented image", "augmented mask"])) :
            axs[i,j].imshow(im)
            axs[i,j].set_axis_off()
            trans_name = trans_name + " :"
            if i+j == 0 :
                title = trans_name + "\n" + title
            if (i!=0) and (j==0) :
                title = trans_name
            if (i!=0) and (j!=0) :
                title = ""
            axs[i,j].set_title(title, fontsize=8)
            

    # main title
    fig.suptitle("Image augmentation tests")

    plt.show()






def testAlbu(gen, trans, idx, cats_colors, trans_name = "") :
    '''
    test augmentation using Albumentation and plot the result for an image and its mask

    parameters :
    ------------
    gen - generator custom class instance
    trans - list of Albumentations transforms
    idx - int : index in generator "gen"
    cats_colors - array of shape (num of classes, 3). RGB colors for each grayscaled class in masks
    trans_name - string : the name of transformation, to put in title. By default : ""
    '''

    # imports 
    import matplotlib.pyplot as plt
    import numpy as np


    # get image and its mask from "gen"
    image = gen[0][0][idx].astype("uint8")
    mask = gen[0][1][idx].astype("uint8")

    # create compose object from "trans" and use it on image and mask
    aug = A.Compose(trans)
    sample = aug(image=image, mask=mask)
    image_aug = sample["image"]
    # for mask, get each pixel label using "argmax" and put it in a channel using "expand_dim"
    mask = cats_colors[np.argmax(mask, axis=2)]
    mask_aug = cats_colors[np.argmax(sample["mask"], axis=2)]

    # create figure
    fig, axs = plt.subplots(2,2,figsize=(14,7))
    axs = axs.flatten()

    # imshow and titles
    for i,(im,title) in enumerate(zip([image, mask, image_aug, mask_aug],["image", "mask", "augmented image", "augmented mask"])) :
        axs[i].imshow(im)
        axs[i].set_axis_off()
        axs[i].set_title(title)

    # main title
    fig.suptitle("Image augmentation test : "+trans_name)





def testModel(model, test_gen, n_images, cats_colors, random_state=16, save_path=None) :
    '''
    test a segmentation model. Display the image, the mask and the predicted mask

    parameters :
    ------------
    model - segmentation model
    test_gen - generator custom class instance
    n_images - int
    random_state - int : random seed. By default : 16
    save_path - string : to save figure. By default, None
    '''

    # imports 
    import matplotlib.pyplot as plt
    import numpy as np
    # random seed
    np.random.seed(seed=random_state)

    # pick one batch
    some_batch_index = np.random.choice(np.arange(max(1,len(test_gen))), size = 1)[0]
    some_batch = test_gen[some_batch_index]
    # pick n_images idx from this batch
    some_batch_examples_idx = np.random.choice(np.arange(len(some_batch[0])), size = n_images)
    some_images = np.array([some_batch[0][i] for i in some_batch_examples_idx])
    some_masks = np.array([some_batch[1][i] for i in some_batch_examples_idx])

    # get class label for each pixel then put them in a channel
    some_masks = cats_colors[np.argmax(some_masks, axis=-1)]


    preds = model.predict(some_images)
    preds = cats_colors[np.argmax(preds, axis=-1)]

    # create figure
    fig, axs = plt.subplots(n_images,3,figsize=(14,7*n_images/3))

    # imshow 
    for i in range(n_images) :
        for j, images_array in enumerate([some_images, some_masks, preds]) :
            axs[i,j].imshow(images_array[i])
            axs[i,j].set_axis_off()

    # titles
    axs[0,0].set_title("Images")
    axs[0,1].set_title("Masks")
    axs[0,2].set_title("Predicted masks")
    fig.suptitle("Segmentation model, visualize predictions")

    plt.savefig(save_path, format = "png")