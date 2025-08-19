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
        self.left = window.left + 4
        self.top = window.top + 24
        self.right = window.right - 4
        self.bottom = window.bottom - 4

        self.height = self.bottom - self.top
        self.width = self.right - self.left

        self.region = (self.left, self.top, self.right, self.bottom)

    def normalize_mouse_pos(self, pos):
        """
        normalize pixel position on the screen depending on the size and location of the game window between 0, 1
        WARNING: if you move the window being captured after execution is started, this will
        return incorrect coordinates, because the window position is only calculated in the __init__ constructor.
        :param pos: (x, y)
        :return: normalized coordinates, -1, -1 if outside game window
        """

        x = pos[0]
        y = pos[1]

        normalized_x = (x - self.left) / self.width
        normalized_y = (y - self.top) / self.height

        if normalized_y < 0 or normalized_y > 1 or normalized_x < 0 or normalized_x > 1:
            normalized_y = -1
            normalized_x = -1

        return normalized_x, normalized_y

    def denormalize_mouse_pos(self, normalized_pos):
        """
        Given mouse position normalized to screen location, translate it back to absolute positioning
        :param normalized_pos: (x, y)
        :return: pos(x, y)
        """

        x = normalized_pos[0]
        y = normalized_pos[1]

        if 0 <= x <= 1 and 0 <= y <= 1:
            offset_x = x * self.width
            offset_y = y * self.height

            return self.left + offset_x, self.top + offset_y

        return -1, -1
