import numpy as np
import cv2 as cv


def prune_data(image_file, output_file):
    """
    Remove data points where the mouse coordinate is -1, -1

    :param image_file: Name of image file
    :param output_file: Name of output file
    :return:
    """
    indexes = []
    image_data = list(np.load(image_file))
    output_data = list(np.load(output_file))

    for i in range(len(image_data)):

        output = output_data[i]

        if tuple(output) == (-1, -1):
            indexes.append(i)

    for index in sorted(indexes, reverse=True):
        print(f"Deleting index with value {output_data[index]}")
        del image_data[index]
        del output_data[index]

    np.save(image_file, image_data)
    np.save(output_file, output_data)


def process_image(image_file):
    """
    Convert to grayscale and add circle detection

    :param image_file: Name of image file
    :return: List of processed images
    """
    images = np.load(image_file)

    processed_images = []

    for screenshot in images:

        screenshot = cv.resize(screenshot, (800, 600), interpolation=cv.INTER_AREA)

        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)

        screenshot = cv.medianBlur(screenshot, 5)

        circles = cv.HoughCircles(screenshot, cv.HOUGH_GRADIENT, 1, 10,
                                  param1=75, param2=50,
                                  minRadius=30, maxRadius=45)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(screenshot, center, 1, (0, 100, 100), 3)

        screenshot = cv.resize(screenshot, (224, 224), interpolation=cv.INTER_AREA)
        processed_images.append(screenshot)

    return processed_images
