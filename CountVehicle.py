import time
from random import randint

import cv2
import numpy as np

import validator
from BGSTypes import getBGSubtractor
from Centroid import getCentroid
from Filter import getFilter
from main import *

# ______________________________________________________________

LINE_IN_COLOR = (64, 255, 0)
LINE_OUT_COLOR = (0, 0, 255)
BOUNDING_BOX_COLOR = (255, 128, 0)
TRACKER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
CENTROID_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
TEXT_POSITION_BGS = (10, 50)
TEXT_POSITION_COUNT_CARS = (10, 100)
TEXT_POSITION_COUNT_TRUCKS = (10, 150)
TEXT_SIZE = 1.2
FONT = cv2.FONT_HERSHEY_SIMPLEX
SAVE_IMAGE = False
IMAGE_DIR = "./veiculos"
VIDEO_SOURCE = "Videos/Cars.mp4"
VIDEO_OUT = "Videos/Results/result_traffic.avi"

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[2]


# ______________________________________________________________



def save_frame(frame, file_name, flip=True):
    if flip:  # BGR -> RGB
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)


cap = cv2.VideoCapture(VIDEO_SOURCE)
hasFrame, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer_video = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

# ROI
bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox

frameArea = h2 * w2
minArea = int(frameArea / 250)
maxArea = 15000

line_IN = int(h1)
line_OUT = int(h2 - 20)

DOWN_limit = int(h1 / 4)
print('Down IN limit Y', str(DOWN_limit))
print('Down OUT limit Y', str(line_OUT))

bg_subtractor = getBGSubtractor(BGS_TYPE)

def countVehicle(qtd_small_cars, qtd_big_cars):
    frame_number = -1
    cnt_small_cars, cnt_big_cars = qtd_small_cars, qtd_big_cars
    objects = []
    max_p_age = 2
    pid = 1

    while (cap.isOpened()):

        ok, frameRoi = cap.read()

        if not ok:
            print('ERRO')
            break

        roi: object = frameRoi[h1:h1 + h2, w1:w1 + w2]

        for i in objects:
            i.age_one()

        frame_number += 1
        bg_mask = bg_subtractor.apply(roi)
        bg_mask = getFilter(bg_mask, 'combine')

        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # contagem de carros pequenos

            if area > minArea and area <= maxArea:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]

                new = True

                cv2.rectangle(roi, (x, y), (x + 50, y - 13), TRACKER_COLOR, -1)
                cv2.putText(roi, 'CAR', (x, y - 2), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(DOWN_limit):
                            cnt_small_cars += 1
                            print('ID:', i.getId(), ' Passou pela via em: ', time.strftime('%c'))
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + 'car_DOWN_%04d.png' % frame_number)
                        break

                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_OUT:
                            i.setDone()
                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i
                if new:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1
                cv2.circle(roi, (cx, cy), 5, CENTROID_COLOR, -1)

            # contagem de carros grandes

            elif area > maxArea:

                x, y, w, h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]

                new = True

                cv2.rectangle(roi, (x, y), (x + 50, y - 13), TRACKER_COLOR, -1)
                cv2.putText(roi, 'CAR', (x, y - 2), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(DOWN_limit):
                            cnt_small_cars += 1
                            print('ID:', i.getId(), ' Passou pela via em: ', time.strftime('%c'))
                            if SAVE_IMAGE:
                                save_frame(roi, IMAGE_DIR + 'car_DOWN_%04d.png' % frame_number)
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_OUT:
                            i.setDone()
                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i
                if new:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1

                cv2.circle(roi, (cx, cy), 5, CENTROID_COLOR, -1)

        for i in objects:
            cv2.putText(roi, str(i.getId()), (i.getX(), i.getY()), FONT, 0.3, TEXT_COLOR, 1, cv2.LINE_AA)

        frameRoi = cv2.line(frameRoi, (w1, line_IN), (w1 + w2, line_IN), LINE_IN_COLOR, 2)

        frameRoi = cv2.line(frameRoi, (w1, h1 + line_OUT), (w1 + w2, h1 + line_OUT), LINE_OUT_COLOR, 2)


        cv2.imshow('Frame', frameRoi)
        cv2.imshow('BG mask', bg_mask)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    return cnt_small_cars+cnt_big_cars

    cap.release()
    cv2.destroyAllWindows()



x = countVehicle(0, 0)

print(x)
#