import cv2
import os


def make_video():
    # windows:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

    for i in range(0, 14000, 100):
        img_path = f"qtable_charts/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video()