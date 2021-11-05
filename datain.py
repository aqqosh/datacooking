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
    def __init__(self, source_img, dest_img, param):
        self.source_img = source_img
        self.dest_img = dest_img
        self.param = param
    def ShowSource(self):
        plt.imshow(cv2.cvtColor(self.source_img, cv2.COLOR_BGR2RGB))
        plt.show()
    def ShowDest(self):
        plt.imshow(cv2.cvtColor(self.dest_img, cv2.COLOR_BGR2RGB))
        plt.show()
    def ResizePair(self, size):
        self.source_img = cv2.resize(self.source_img, size, interpolation = cv2.INTER_AREA)
        self.dest_img = cv2.resize(self.dest_img, size, interpolation = cv2.INTER_AREA)
    def ShowPair(self):
        if self.source_img.size != self.dest_img.size:
            self.ResizePair(size=(512, 512))
        pair_img = np.concatenate((self.source_img, self.dest_img), axis=1)
        plt.imshow(cv2.cvtColor(pair_img, cv2.COLOR_BGR2RGB))
        plt.show()
    def DetectHuman(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        #gray = cv2.cvtColor(self.source_img, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(self.source_img, winStride=(8,8))
        self.boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in self.boxes:
            cv2.rectangle(self.source_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
        for (xA, yA, xB, yB) in self.boxes:
            cv2.rectangle(self.dest_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    def Crop(self):
        human = self.boxes[0]
        xA, yA, xB, yB = human
        self.source_img = self.source_img[yA:yB, xA:xB]
        self.dest_img = self.dest_img[yA:yB, xA:xB]
    def Transformation(self, transform, execute_all=False):
        if execute_all:
            transformed = transform(image=self.source_img, mask=self.dest_img)
            return Pair(transformed['image'], 
                        transformed['mask'], param="artificial")
        if self.param == "natural":
            transformed = transform(image=self.source_img, mask=self.dest_img)
            return Pair(transformed['image'], 
                        transformed['mask'], param="artificial")
        return None


    

class DataFolder:
    def __init__(self, path: str):
        self.path = path
        self.source_path = path + "/source/"
        self.dest_path = path + "/dest/"
        self.imgs = []
    def GetSourceNames(self):
        self.source = os.listdir(self.source_path)
        return self.source
    def GetDestNames(self):
        self.dest = os.listdir(self.dest_path)
        return self.dest
    def CheckNames(self):
        for source_name, dest_name in zip(self.source, self.dest):
            if source_name != dest_name:
                return False
            return True
    def CheckCount(self):
        if len(self.source) != len(self.dest):
            raise DataPairingError("Amoung of files in data \
                                    folders doesn't match")
            return False
        return True           
    def LoadPairs(self):
        if self.CheckCount():
            for source_name, dest_name in zip(self.source, self.dest):
                source_img = cv2.imread(self.source_path + source_name, cv2.COLOR_BGR2RGB)
                dest_img = cv2.imread(self.dest_path + dest_name, cv2.COLOR_BGR2RGB)
                new_pair = Pair(source_img, dest_img, param="natural")
                self.imgs.append(new_pair)
            print(str(len(self.imgs)) + " pair of images loaded")
            return True
        return False
    def UnloadPairs(self):
        del self.imgs
        self.imgs = []
    def ResizeAllImages(self, size=(512, 512)):
        if self.imgs != []:
            for pair in self.imgs:
                pair.ResizePair(size)
            print("All pairs were resized successfully")
    def DetectAllHumans(self):
        if self.imgs != []:
            for pair in self.imgs:
                pair.DetectHuman()
            print("All humans were found successfully")
    def CropAllPairs(self):
        if self.imgs != []:
            for pair in self.imgs:
                pair.Crop()
            print("All Crops were done successfully")
    def Save(self, path, type):
        if not os.path.exists(path):
            os.mkdir(path) 
        if type == "source":
            for pair, name in zip(self.imgs, self.source):
                cv2.imwrite(path + name, pair.source_img)
        else:
            print("Types except of source doesn't support")
    def MassiveTransform(self, transform, execute_all=False):
        for pair in self.imgs:
            new_pair = pair.Transformation(transform=transform, 
                                        execute_all=execute_all)
            if new_pair != None:
                self.imgs.append(new_pair)

    
data_path = "datacooking/data"

test_data_folder = DataFolder(data_path)
test_data_folder.GetSourceNames()
test_data_folder.GetDestNames()
test_data_folder.CheckNames()
test_data_folder.CheckCount()
test_data_folder.UnloadPairs()
test_data_folder.LoadPairs()
test_data_folder.ResizeAllImages(size = (1920, 1080))

#test_images_pair = test_data_folder.imgs[0]
#test_images_pair.DetectHuman()
#test_images_pair.Crop()
#test_images_pair.ShowPair()

test_data_folder.DetectAllHumans()
test_data_folder.CropAllPairs()
#test_data_folder.Save(path="data_cooking/data/Crop_source/", type="source")

transform = A.Compose([
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=0.2),
])

test_data_folder.MassiveTransform(transform=transform, execute_all=False)
test_data_folder.imgs[-3].ShowPair()