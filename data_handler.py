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
    #picklefile = "cloi_storage/_data/test_video/boxes.pickle"
    #data_source_names = "cloi_storage/_data/test_video/output"
    #data_dest_names = "cloi_storage/_data/test_video/dest"
    #source_out = "cloi_storage/_data/test_video/source_out"
    #dest_out = "cloi_storage/_data/test_video/dest_out"
    #img_size = (512, 512)
    
    data_source_names = "cloi_storage/_data/test_video/yerena_out"
    data_dest_names = "cloi_storage/_data/test_video/yerena_out"
    source_out = "cloi_storage/_data/test_video/test_data/test_B"
    dest_out = "cloi_storage/_data/test_video/test_data/test_B"
    
    img_size = (512, 512)

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
    
    @staticmethod
    def get_bbox(img):
        a = np.where(img != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return bbox
        
    @classmethod
    def execute(cls):
        for i, image_name in enumerate(cls.source):
            #TODO separate method
            source_img = cv2.imread("{}/{}".format(cls.data_source_names, image_name), cv2.COLOR_BGR2RGB)
            dest_img = cv2.imread("{}/{}".format(cls.data_dest_names, image_name), cv2.COLOR_BGR2RGB) 
            print("Loaded source image shape: {}".format(source_img.shape))
            #TODO separate method
            bbox_coordinates = cls.get_bbox(source_img)
            xA, yA, xB, yB = bbox_coordinates
            print("Calculated bbox for image: {}".format(bbox_coordinates))
            #TODO separate method
            source_img = source_img[xA:yA, xB:yB]
            dest_img = dest_img[xA:yA, xB:yB]
            #TODO separate method
            #resize to parametric image size
            source_img = cv2.resize(source_img, cls.img_size, interpolation = cv2.INTER_AREA)
            dest_img = cv2.resize(dest_img, cls.img_size, interpolation = cv2.INTER_AREA)
            #TODO separate method
            #save to directory
            print("Saving {} \n {}/{}".format(image_name, i + 1, len(cls.source)))
            cv2.imwrite("{}/{}".format(cls.source_out, image_name), source_img)
            cv2.imwrite("{}/{}".format(cls.dest_out, image_name), dest_img)
        
class InferenceData(Data):
    """
    1. Посылает пришедшие на вход данные в DensePose
    2. Depsepose прогоняется через препроцессинг,
        сохраняются коодинаты bboxа и size оригинала 
        для каждого изображения
    3. Данные отправляются в pix2pixHD
    4. В результате затирается зеленый фон, сглаживается
    5. Результ ресайзится и пастится на оригинальное
        изображение
    6. *Вырезаются руки-ноги
    7. *Применяется LAMA
    """
    @classmethod
    def get_densepose_data(cls):
        cls.bboxes = {}
        cls.sizes = {}
        
        cls.original_images = {}
        cls.processed_images = {}
    
    @classmethod
    def get_pix2pixHD_data(cls):
        cls.pix2pix_images = {}
        
    @classmethod
    def postprocessing(cls):
        cls.result_imges = {}
    
    @classmethod
    def execute():
        pass
    

my_data = TrainingData()
my_data.get_methadata()
my_data.get_data_imgs()
my_data.execute()