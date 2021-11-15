import os

path = 'C:\\Users\\Elizaveta Koroleva\\Desktop\\Courses\\datacooking\\data\\for_model\\dest'
files = os.listdir(path)


for index, file in enumerate(files):
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