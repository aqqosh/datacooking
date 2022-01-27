import os
import glob

path = r'C:\Users\Elizaveta Koroleva\Desktop\ProdData - Copy\source_pose'
files = os.listdir(path)

#files = glob.iglob(path + '/*.' + "jpg")


for index, file in enumerate(files):
    #print(str(file)[9:-5])
    index += 372
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))



"""
path = 'C:\\Users\\Elizaveta Koroleva\\Desktop\\Courses\\datacooking\\data\\for_model\\source'
files = os.listdir(path)


for file in files:
    if file.endswith("IUV.png"):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([file[:-8], '.jpg'])))
    else:
        os.remove(os.path.join(path, file))
"""

path = 'C:\\Users\\Elizaveta Koroleva\\Desktop\\Courses\\datacooking\\data\\densepose\\file\\content\\DensePose\\DensePoseData\\output_1'
files = os.listdir(path)


for file in files:
    if file.endswith("IUV.png"):
        os.rename(os.path.join(path, file), os.path.join(path, file.replace("IUV.png", ".jpg")))
    else:
        os.remove(os.path.join(path, file))

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, file.replace("_.jpg", ".jpg")))