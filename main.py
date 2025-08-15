import os
import random

import torch

import cv2
import torchvision
from torch import nn, optim
import dxcam

import windowcapture
from time import time

import pyautogui
import pydirectinput
import keyboard
import cv2 as cv
import numpy as np

from dataset import SCREEN_WIDTH, SCREEN_HEIGHT
from train_data import num_classes, device, learning_rate, load_checkpoint
from windowcapture import WindowCapture
from torchvision import transforms
from windowcapture2 import get_window

image_file = "image_data.npy"
abs_file = "abs_data.npy"
rel_file = "rel_data.npy"


def get_data(image_data, abs_data, rel_data, screenshot):
    # Playing at fullscreen, otherwise use windowcapture.get_screen_position
    (abs_x, abs_y) = pyautogui.position()

    (rel_x, rel_y) = (0, 0)

    if abs_data:
        # Subtract previous position from current position
        rel_x = abs_x - abs_data[-1][0]
        rel_y = abs_y - abs_data[-1][1]



    image_data.append(screenshot)
    abs_data.append((abs_x, abs_y))
    rel_data.append((rel_x, rel_y))



    if len(image_data) % 500 == 0:
        np.save(image_file, image_data)
        np.save(abs_file, abs_data)
        np.save(rel_file, rel_data)
        print("data saved")



if __name__ == '__main__':

    wincap = WindowCapture("osu!")

    loop_time = time()


    # if os.path.isfile(image_file):
    #     print("File found")
    #     image_data = list(np.load(image_file))
    #     abs_data = list(np.load(abs_file))
    #     rel_data = list(np.load(rel_file))
    # else:
    #     print("File not found")
    #     image_data = []
    #     abs_data = []
    #     rel_data = []

    model = torchvision.models.resnet50(weights=True)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)

    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    load_checkpoint(model, optimizer, torch.load("my_checkpoint.pth.tar"))

    transform = transforms.ToTensor()

    model.eval()

    camera = dxcam.create()
    camera.start(target_fps=0, video_mode=True)

    while True:

        # screenshot = wincap.get_screenshot()

        screenshot = camera.get_latest_frame()


        if screenshot is not None:
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            cv.imshow("test", screenshot)

            # 256 x 144 images
            screenshot = cv2.resize(screenshot, (256, 144), interpolation=cv2.INTER_AREA)

            # get_data(image_data, abs_data, rel_data)

            tensor_image = transform(screenshot)

            tensor_image = tensor_image.reshape((1,) + tensor_image.shape)

            prediction = model(tensor_image.to(device))

            test = prediction.tolist()

            # Returns width, height

            (x, y) = test[0]

            # print("Prediction: ", x, y)

            x *= SCREEN_WIDTH
            y *= SCREEN_HEIGHT

            pyautogui.moveTo(x, y)

            # Show image
            # cv.imshow('Computer Vision', screenshot)



            # debug the loop rate
            print('FPS {}'.format(1 / (time() - loop_time)))

            loop_time = time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break

        if keyboard.is_pressed('p'):
            break
