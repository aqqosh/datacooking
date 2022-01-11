import os
import cv2
import pickle
import argparse
import numpy as np

class Data(object):
    @classmethod
    def get_methadata():
        pass
    
    @classmethod
    def get_data_imgs(cls):
        pass
    
    @classmethod
    def execute():
        pass
    

class TrainingData(Data):
    picklefile = "cloi_storage/_data/test_video/boxes.pickle"
    data_source_names = "cloi_storage/_data/test_video/output"
    data_dest_names = "cloi_storage/_data/test_video/dest"
    source_out = "cloi_storage/_data/test_video/source_out"
    dest_out = "cloi_storage/_data/test_video/dest_out"
    
    img_size = (512, 512)
    
    @classmethod
    def get_methadata(cls):
        with open(cls.picklefile, 'rb') as handle:
            cls.methadata = pickle.load(handle)

    @classmethod
    def get_data_imgs(cls):
        cls.source = os.listdir(cls.data_source_names)
        cls.dest = os.listdir(cls.data_dest_names)
        
        if cls.check_names(cls.source, cls.dest):
            return cls.source
        
        return []
            
    @staticmethod
    def check_names(list1, list2):
        for a, b in zip(list1, list2):
            if a != b:
                return False
        return True
        
    @classmethod
    def execute(cls):
        def bbox1(img):
            a = np.where(img != 0)
            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            return bbox
        
        for i, image_name in enumerate(cls.source):
            metha = cls.methadata[image_name][0]
            source_img = cv2.imread("{}/{}".format(cls.data_source_names, image_name), cv2.COLOR_BGR2RGB)
            dest_img = cv2.imread("{}/{}".format(cls.data_dest_names, image_name), cv2.COLOR_BGR2RGB) 
            print(source_img.shape)
            
            #round
            #metha = list(map(lambda x: round(x), metha))
            #print(metha)
            #crop by metha = coordinames
            #xA, yA, xB, yB = metha
            #source_img = cv2.circle(source_img, (xB, xA), radius=5, color=(0, 0, 255), thickness=-1)
            #source_img = cv2.circle(source_img, (yB, yA), radius=5, color=(0, 0, 255), thickness=-1)
            #source_img = source_img[yA: yA + yB, xA : xA + yA]
            #dest_img = dest_img[yA: yA + yB, xA : xA + yA]
            #cv2.imshow("source", source_img)
            #cv2.imshow("dest", dest_img)
            
            dat = bbox1(source_img)
            xA, yA, xB, yB = dat
            print(dat)

            source_img = source_img[xA:yA, xB:yB]
            dest_img = dest_img[xA:yA, xB:yB]

            #resize to parametric image size
            source_img = cv2.resize(source_img, cls.img_size, interpolation = cv2.INTER_AREA)
            dest_img = cv2.resize(dest_img, cls.img_size, interpolation = cv2.INTER_AREA)
            
            #save to directory
            print("Saving {} \n {}/{}".format(image_name, i, len(cls.methadata)))
            cv2.imwrite("{}/{}".format(cls.source_out, image_name), source_img)
            cv2.imwrite("{}/{}".format(cls.dest_out, image_name), dest_img)
        
class InferenceData(Data):
    @classmethod
    def get_methadata():
        pass
    
    @classmethod
    def get_data_imgs(cls):
        pass
    
    @classmethod
    def execute():
        pass
    

my_data = TrainingData()
my_data.get_methadata()
my_data.get_data_imgs()
my_data.execute()