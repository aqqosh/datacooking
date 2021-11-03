import cv2
import os
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DataPairingError(Error):
    """When data in pairs doesn't match"""
    pass

class Pair:
    def __init__(self, source_img, dest_img):
        self.source_img = source_img
        self.dest_img = dest_img
    def show_source(self):
        plt.imshow(cv2.cvtColor(self.source_img, cv2.COLOR_BGR2RGB))
        plt.show()
    def show_dest(self):
        plt.imshow(cv2.cvtColor(self.dest_img, cv2.COLOR_BGR2RGB))
        plt.show()
    def resize_pair(self, size):
        self.source_img = cv2.resize(self.source_img, size, interpolation = cv2.INTER_AREA)
        self.dest_img = cv2.resize(self.dest_img, size, interpolation = cv2.INTER_AREA)
    def show_pair(self):
        if self.source_img.size != self.dest_img.size:
            self.resize_pair(size=(512, 512))
        pair_img = np.concatenate((self.source_img, self.dest_img), axis=1)
        plt.imshow(cv2.cvtColor(pair_img, cv2.COLOR_BGR2RGB))
        plt.show()
    def detect_human(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        #gray = cv2.cvtColor(self.source_img, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(self.source_img, winStride=(8,8))
        self.boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in self.boxes:
            cv2.rectangle(self.source_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
        for (xA, yA, xB, yB) in self.boxes:
            cv2.rectangle(self.dest_img, (xA, yA), (xB, yB), (0, 255, 0), 2)

    

class DataFolder:
    def __init__(self, path: str):
        self.path = path
        self.source_path = path + "/source/"
        self.dest_path = path + "/dest/"
        self.imgs = []
    def get_source_names(self):
        self.source = os.listdir(self.source_path)
        return self.source
    def get_dest_names(self):
        self.dest = os.listdir(self.dest_path)
        return self.dest
    def check_names(self):
        for source_name, dest_name in zip(self.source, self.dest):
            if source_name != dest_name:
                return False
            return True
    def check_count(self):
        if len(self.source) != len(self.dest):
            raise DataPairingError("Amoung of files in data \
                                    folders doesn't match")
            return False
        return True           
    def load_pairs(self):
        if self.check_count():
            for source_name, dest_name in zip(self.source, self.dest):
                source_img = cv2.imread(self.source_path + source_name, cv2.COLOR_BGR2RGB)
                dest_img = cv2.imread(self.dest_path + dest_name, cv2.COLOR_BGR2RGB)
                new_pair = Pair(source_img, dest_img)
                self.imgs.append(new_pair)
            print(str(len(self.imgs)) + " pair of images loaded")
            return True
        return False
    def unload_pairs(self):
        del self.imgs
        self.imgs = []
    def resize_all_images(self, size=(512, 512)):
        if self.imgs != []:
            for pair in self.imgs:
                pair.resize_pair(size)
            print("All pairs were resized successfully")
            

    
data_path = "datacooking/data"

test_data_folder = DataFolder(data_path)
test_data_folder.get_source_names()
test_data_folder.get_dest_names()
test_data_folder.check_names()
test_data_folder.check_count()
test_data_folder.unload_pairs()
test_data_folder.load_pairs()
test_data_folder.resize_all_images(size = (1920, 1080))

test_images_pair = test_data_folder.imgs[0]
#test_images_pair.resize_pair(size=(512, 512))
test_images_pair.detect_human()
test_images_pair.show_pair()
#test_images_pair.show_source()




plt.imshow(frame, cmap='gray')
plt.show()

