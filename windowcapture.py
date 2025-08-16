import numpy as np
import win32gui, win32ui, win32con
from ctypes import windll

import dxcam
import pygetwindow


class WindowCapture:

    def __init__(self, window_name=None):
        # find the handle for the window we want to capture

        if not window_name:
            window = pygetwindow.getActiveWindow()

        else:
            # get first window that starts with window name
            windows = pygetwindow.getWindowsWithTitle(window_name)
            if not windows:
                raise Exception('Window not found: {}'.format(window_name))
            window = windows[0]

        # remove top bar and black bars from image
        self.x = window.left + 4
        self.y = window.top + 24
        self.right = window.right - 4
        self.bottom = window.bottom - 4

        self.region = (self.x, self.y, self.right, self.bottom)

    def normalize_mouse_pos(self, pos):
        """
        normalize pixel position on the screen depending on the size of the game window
        WARNING: if you move the window being captured after execution is started, this will
        return incorrect coordinates, because the window position is only calculated in the __init__ constructor.
        :param pos: (x, y)
        :return:
        """
        return pos[0] + self.x, pos[1] + self.y

    def denormalize_mouse_pos(self, normalized_pos):
        """
        Given mouse position normalized to screen location, translate it back to
        :param normalized_pos:  (x, y)
        :return: pos(x, y)1
        """
        return