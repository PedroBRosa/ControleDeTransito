import numpy as np
import cv2

VIDEO_SOURCE = 'Videos/Cars.mp4'
VIDEO_OUT = 'Videos/Results/filtragem_mediana_tempora.avi'


cap = cv2.VideoCapture(VIDEO_SOURCE)
hasFrame, frame = cap.read()
# print(hasFrame, frame.shape)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1], frame.shape[0]), True)


