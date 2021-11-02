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

class DataFolder:
    def __init__(self, path: str):
        self.path = path
        self.source_path = path + "/source/"
        self.dest_path = path + "/dest/"
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
    """                     
    def load_pairs(self):
        for source_name, dest_name in zip(self.source, self.dest):
    """

    
data_path = "datacooking/data"
test_data_folder = DataFolder(data_path)
test_data_folder.get_source_names()
test_data_folder.get_dest_names()
test_data_folder.check_names()

class Pair:
    def 

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

frame = cv2.imread("datacooking/data/source/0049.jpg0040.jpg")
#frame = cv2.resize(frame, (640, 480))
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

for (xA, yA, xB, yB) in boxes:
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

plt.imshow(frame, cmap='gray')

while(True):
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

