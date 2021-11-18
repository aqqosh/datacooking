import cv2
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

transform1 = A.Compose([A.HorizontalFlip(p=1), ])
transform2 = A.Compose([A.Affine(rotate=(-15, 15), p=1, mode=cv2.BORDER_REFLECT)])
transform3 = A.Compose([A.Affine(scale=(1.1, 1.9), p=1)])

transform_result_path_1 = test_data_folder.TransformWithoutLoading(
                                        transform=transform1, 
                                        first_transform=True)

test_data_folder = DataFolder(transform_result_path_1)
transform_result_path_2 = test_data_folder.TransformWithoutLoading(transform=transform2)

test_data_folder = DataFolder(transform_result_path_2)
transform_result_path_3 = test_data_folder.TransformWithoutLoading(transform=transform3)
