import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DataPairingError(Error):
    """When data in pairs doesn't match"""
    pass

class Pair(object):
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


class DataFolder(object):
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

    def CheckFolderAmount(self):
        if len(self.source) != len(self.dest):
            raise DataPairingError("Amoung of files in data \
                                    folders doesn't match")
            return False
        return True         

    def LoadPairs(self):
        if self.CheckFolderAmount():
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

    def Save(self, path):
        if self.CheckPathExists(path=path):
            self.CreateSubfolders(path=path) 
        for pair, name in zip(self.imgs, self.source):
            cv2.imwrite(path + "source/" + name, pair.source_img)
            cv2.imwrite(path + "dest/" + name, pair.dest_img)

    def MassiveTransform(self, transform, execute_all=False):
        for pair in self.imgs:
            new_pair = pair.Transformation(transform=transform, 
                                        execute_all=execute_all)
            if new_pair != None:
                self.imgs.append(new_pair)
                
    def CheckAmount(self):
        return len(self.imgs)

    def CheckPathExists(self, path):
        return True if os.path.exists else False

    def CreateSubfolders(self, path):
        os.mkdir(path)
        os.mkdir(path + "source/")
        os.mkdir(path + "dest/")

    def TransformWithoutLoading(self, transform, save_dir="/output/"):
        if self.CheckFolderAmount():
            for source_name, dest_name in zip(self.source, self.dest):
                source_img = cv2.imread(self.source_path + source_name, cv2.COLOR_BGR2RGB)
                dest_img = cv2.imread(self.dest_path + dest_name, cv2.COLOR_BGR2RGB)

                pair = Pair(source_img, dest_img, param="natural")
                new_pair = pair.Transformation(transform=transform, execute_all=True)

                if self.CheckPathExists(path=save_dir):
                    self.CreateSubfolders(path=save_dir)

                cv2.imwrite(save_dir + "source/" + source_name + "t", new_pair.source_img)
                cv2.imwrite(save_dir + "dest/" + dest_name + "t", new_pair.dest_img)

            print(str(len(self.imgs)) + " pair of images transformed")
            return True
        return False