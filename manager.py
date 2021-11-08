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

import albumentations as A
from datacooking.datain import DataFolder, Pair
    
data_path = "datacooking/data"

test_data_folder = DataFolder(data_path)
test_data_folder.GetSourceNames()
test_data_folder.GetDestNames()
test_data_folder.CheckNames()
test_data_folder.CheckFolderAmount()
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
test_data_folder.CheckAmount()
test_data_folder.Save(path="datacooking/data/cropped_imgs/")

transform1 = A.Compose([
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(p=0.2),
])

test_data_folder.MassiveTransform(transform=transform1, execute_all=False)

test_data_folder.CheckAmount()

transform2 = A.Compose([
    A.Affine(scale=(0.05, 0.25), rotate=(-20, 20), p=1)
])
test_data_folder.MassiveTransform(transform=transform2, execute_all=True)

#test_data_folder.imgs[-3].ShowPair()
#dir(test_data_folder.imgs[-3])