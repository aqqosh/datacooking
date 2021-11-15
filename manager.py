# by hands:
# add several "in-the-wild" examples 
# add several complicated "in-the-wild" examples
# remove bad examples

# data in:
# data pairs loading +
# cutting by human-size +
# source masks recieving 
# pairs cutting in bottom/left/right 
# pairs streatching (top-bottom, left-right)
# pairs rotating
# pairs resize to 512x512 // result should be: various shapes of person

# data out:
# resize source - resize costume back
# insert costume to the source
# remove green background
# blend edges of the costume

# Трансформировать данные так, чтобы на изображениях размером 512
# человек выглядел и толстым, и худым, и высоким, и низким.

import albumentations as A
from datacooking.datain import DataFolder, Pair
    
data_path = "datacooking/data"

test_data_folder = DataFolder(data_path)
test_data_folder.GetSourceNames()
test_data_folder.GetDestNames()
test_data_folder.CheckNames()
test_data_folder.CheckFolderAmount()
test_data_folder.UnloadPairs()
#test_data_folder.LoadPairs()
#test_data_folder.ResizeAllImages(size = (1920, 1080))

#test_images_pair = test_data_folder.imgs[0]
#test_images_pair.DetectHuman()
#test_images_pair.Crop()
#test_images_pair.ShowPair()

#test_data_folder.DetectAllHumans()
#test_data_folder.CropAllPairs()
#test_data_folder.CheckAmount()
#test_data_folder.Save(path="datacooking/data/cropped_imgs_v2/")

transform1 = A.Compose([
    A.HorizontalFlip(p=1),
])

#test_data_folder.MassiveTransform(transform=transform1, execute_all=False)
#test_data_folder.CheckAmount()

test_data_folder.TransformWithoutLoading(transform=transform1, save_dir="datacooking/data/transformed_imgs_v2/", first_transform=True)

test_data_folder = DataFolder("datacooking/data/transformed_imgs_v2/")
test_data_folder.GetSourceNames()
test_data_folder.GetDestNames()
test_data_folder.CheckNames()
test_data_folder.CheckFolderAmount()

transform2 = A.Compose([
    A.Affine(rotate=(-15, 15), p=1, mode=cv2.BORDER_REFLECT)
])
test_data_folder.TransformWithoutLoading(transform=transform2, 
                                        save_dir="datacooking/data/transformed_imgs_v2_2/",
                                        first_transform = False)

test_data_folder = DataFolder("datacooking/data/transformed_imgs_v2_2")
test_data_folder.GetSourceNames()
test_data_folder.GetDestNames()
test_data_folder.CheckNames()
test_data_folder.CheckFolderAmount()

transform3 = A.Compose([
    A.Affine(scale=(1.1, 1.9), p=1)
])

test_data_folder.TransformWithoutLoading(transform=transform3, 
                                        save_dir="datacooking/data/transformed_imgs_v2_3/",
                                        first_transform = False)

#test_data_folder.imgs[-3].ShowPair()
#dir(test_data_folder.imgs[-3])

#import os
#os.path.exists("datacooking/data/transformed_imgs/")