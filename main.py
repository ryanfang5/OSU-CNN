import os

import sys

from pynput.mouse import Controller


import torch
from torch import optim
import dxcam

import time

import keyboard
import cv2 as cv
import numpy as np

from train_data import device, learning_rate, load_checkpoint
from windowcapture import WindowCapture
from torchvision import transforms

from OSU_model import OSUModel

image_file = "image_data.npy"
output_file = "normalized_coords.npy"

load_model = 1

model_name = f'osu_model_{load_model}.pth.tar'

gather_data = False
debug = False


def get_data(image_data, output_data, screenshot, wincap, step):

    (normalized_x, normalized_y) = wincap.normalize_mouse_pos(mouse.position)

    if 0 <= normalized_x <= 1 and 0 <= normalized_y <= 1:

        if step != 0:
            # Output of previous image should be current mouse coordinates
            output_data[-1] = (normalized_x, normalized_y)

        image_data.append(screenshot)
        # Will be updated in next frame, remove samples if coords are -1, -1
        output_data.append((-1, -1))

        if len(image_data) % 500 == 0:
            np.save(image_file, image_data)
            np.save(output_file, output_data)
            print("data saved")

    else:
        sys.exit()


if __name__ == '__main__':

    wincap = WindowCapture("osu!")

    step = 0

    loop_time = time.time()

    if os.path.isfile(image_file):
        print("File found")
        image_data = list(np.load(image_file))
        output_data = list(np.load(output_file))
    else:
        print("File not found")
        image_data = []
        output_data = []

    model = OSUModel()

    model.to(device)

    optimizer = optim.Adam(model.resnet.parameters(), lr=learning_rate)

    load_checkpoint(model, optimizer, torch.load(model_name))

    transform = transforms.ToTensor()

    model.eval()

    camera = dxcam.create()
    camera.start(video_mode=True, region=wincap.region)

    mouse = Controller()

    paused = True

    while True:

        if keyboard.is_pressed("esc"):
            paused = True

        if keyboard.is_pressed("p"):
            paused = False

        if paused:
            # Idle time before unpausing
            time.sleep(0.5)
            step = 0
            print("Paused")

        else:
            screenshot = camera.get_latest_frame()

            if screenshot is not None:

                if debug:
                    cv.imshow("test", screenshot)

                # 224 x 224 images
                screenshot = cv.resize(screenshot, (224, 224), interpolation=cv.INTER_AREA)

                if gather_data:
                    get_data(image_data, output_data, screenshot, wincap, step)
                    step += 1

                else:

                    tensor_image = transform(screenshot)

                    tensor_image = tensor_image.reshape((1,) + tensor_image.shape)

                    prediction = model(tensor_image.to(device))

                    test = prediction.tolist()

                    # Returns width, height

                    (x, y) = test[0]

                    # print("Prediction: ", x, y)

                    (x, y) = wincap.denormalize_mouse_pos((x, y))

                    # move mouse to predicted coordinates

                    mouse.position = (x, y)

                if debug:
                    # debug the loop rate
                    print('FPS {}'.format(1 / (time.time() - loop_time)))

                    loop_time = time.time()

            # press 'q' with the output window focused to exit.
            # waits 1 ms every loop to process key presses
            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break
