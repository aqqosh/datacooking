import cv2
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt

class Data(object):
    
    def flow_from_directory(self, path):
        pass
    
    def flow_from_memory(self, list):
        pass


class DataProcessing(object):
    
    def __init__(self, data):
        self.data = data
        
    def __hash_f(self):
        pass
    
    def __unhash_f(self):
        pass
    
    def apply_densepose(self):
        #send request to flask densepose
        pass
    
    def restructurize(self):
        pass
    
    def apply_pix2pix(self):
        #send request to flask pix2pix
        pass
    
    @staticmethod
    def restore_image(input_img, result_img, original_size=None, bbox=None):
        result_img = remove_green(result_img)
        result_img = jpg2png(result_img)
        
        #TODO: unhash function
        #if not original_size or not bbox:
        #    original_size, bbox = DataProcessing.__unhash_f(image_hash)
        
        result_img = resize(result_img, original_size)
        final_result = blend(input_img, result_img, bbox)
        
        return final_result
    
    def remove_original_garment(self):
        #send request to flask lama
        pass



def remove_green(image, lower=None, upper=None):
    if not lower:
        lower = np.array([0, 0, 0])
    if not upper:
        upper = np.array([255, 150, 255])
        
    mask = cv2.inRange(image, lower, upper)
    image = cv2.bitwise_and(image, image, mask = mask)
    
    return image

def jpg2png(image, lower=None, upper=None):
    if not lower:
        lower = (0, 0, 0)
    if not upper:
        upper = (255, 170, 255)
        
    mask = cv2.inRange(image, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    image[mask == 0] = (255, 255, 255)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
    mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5, 255), out_range=(0, 255))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = mask
    
    return image

def resize(image, original_size):
    resized_image = cv2.resize(image, original_size)
    
    return resized_image

def blend(source_img, dest_img, bbox):
    x, y, _, _ = bbox
    y1, y2 = y, dest_img.shape[0]
    x1, x2 = x, dest_img.shape[1]

    alpha_s = dest_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        source_img[y1:y2, x1:x2, c] = (
            alpha_s * dest_img[:, :, c] + 
            alpha_l * source_img[y1:y2, x1:x2, c]
            )
        
    return source_img

def save_img():
    pass

def show_img():
    pass

#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()


#result_img = jpg2png(result_img)
#plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
#plt.show()

#input_img_path = "datacooking/post/input.jpg"
#result_img_path = "datacooking/post/result.jpg"
#input_img = cv2.imread(input_img_path)
#result_img = cv2.imread(result_img_path)


"""
# augmentation

original
densepose
densepose_segmentation
densepose_skeleton
dest

resized_cropped_original
resized_cropped_densepose_segmentation
resized_cropped_skeleton
resized_cropped_dest

bbox_coordinates
original_size

resized_cropped_green_pix2pix
transparent_pix2pix
blended_result
"""