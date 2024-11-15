
import cv2
import numpy as np

class OpenCVPicLoader:
    def __init__(self):
        pass

    def __call__(self, filePath) -> np.array :
        pic = cv2.imread(str(filePath),cv2.IMREAD_COLOR)
        if pic is None:
            raise FileNotFoundError(f"Image file not found or cannot be read: {filePath}")
        pic = pic[...,::-1] #BGR转RGB
        retpic = np.array(pic)
        return retpic