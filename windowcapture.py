import numpy as np
import win32gui, win32ui, win32con
from ctypes import windll

from collections import deque
class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    x = 0
    y = 0

    def __init__(self, window_name=None):
        # find the handle for the window we want to capture
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        windll.user32.SetProcessDPIAware()

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.x = window_rect[0] + 4
        self.y = window_rect[1] + 24
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        self.circles = deque()
        self.focus = None

    def get_screenshot(self):

        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        bitmap = None
        img = None

        try:
            # All resource acquisition happens here
            hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, self.w, self.h)
            save_dc.SelectObject(bitmap)

            result = windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)

            if result:
                bmpinfo = bitmap.GetInfo()
                bmpstr = bitmap.GetBitmapBits(True)
                img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
                img = np.ascontiguousarray(img)[..., :-1]
            else:
                print(f"Unable to acquire screenshot! PrintWindow returned: {result}")

        except Exception as e:
            print(f"An error occurred during screenshot capture: {e}")
            # img remains None, indicating failure

        finally:
            # Cleanup *always* happens here, regardless of success, failure, or exception
            if save_dc:
                save_dc.DeleteDC()
            if mfc_dc:
                mfc_dc.DeleteDC()
            if bitmap:
                win32gui.DeleteObject(bitmap.GetHandle())
            if hwnd_dc:
                win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.x, pos[1] + self.y)

    def test_screen(self):
        return(self.w, self.h)