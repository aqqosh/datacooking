import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

frame = cv2.imread("datacooking/data/source/0049.jpg0040.jpg")
#frame = cv2.resize(frame, (640, 480))
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

for (xA, yA, xB, yB) in boxes:
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

plt.imshow(frame, cmap='gray')

while(True):
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break