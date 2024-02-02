import cv2
import sys
import os

X_Y_COORDINATE_TEXT = (850, 500)

import sys
def captureImage(v, x):
    c = CaptureImage(v, x)
    return c

class CaptureImage():
    def __init__(self, frame=None, epoch_nano=None):
        self.frame = frame
        self.curr_time = epoch_nano
        self.full_p = os.path.sep.join(['static', "output/capture_frame_blur_{}.png".format(str(self.curr_time).replace(":",''))])
        self.isFrameBlurredWithTextSaved = False

    def _blurred(self):
        self.frame_blurred = cv2.blur(self.frame, (100,100))
        return self.frame_blurred
    
    def _withText(self, top_guess, blurred_img):
        self.frame_blurred_with_text = cv2.putText(
        blurred_img,
        top_guess,
        X_Y_COORDINATE_TEXT, cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (255, 255, 255),
        3,
        cv2.LINE_AA
        )
        return self.frame_blurred_with_text
    
    def _save(self, frame_blurred_with_text):
        cv2.imwrite(self.full_p, frame_blurred_with_text)
        self.isFrameBlurredWithTextSaved = True

if __name__ == '__main__':
    captureImage(sys.argv[1])