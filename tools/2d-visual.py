import os.path

import cv2
import numpy as np
from PIL import Image



camera_intrinsic = {
    "T": [],
    "f": [1696.52023984021, 1691.92302415464],
    "c": [1030.19259698601, 490.845379024388]
}

def get_2d_information():
    image_path = "D:/3Dplantdatas/waxGourd(50test)/JPG/"
    filepath = "D:/3Dplantdatas/waxGourd(50test)/result_label/"
    label_file_list = os.listdir(filepath)
    for file in label_file_list:
        label_2d_path=os.path.join(filepath,file)
        (filename,extension)=os.path.splitext(file)
        image = cv2.imread(image_path+filename+'.jpg')
        with open(label_2d_path, 'r') as f:
            line = f.readline().strip()
            while line:

                line = line.split(" ")
                np.asarray(line)
                line=line[4:8]
                line = np.array(line)
                line = line.astype(float)
                line = line.astype(int)
                line=[[line[2],line[1]],[line[2],line[3]],[line[0],line[3]],[line[0],line[1]]]

            #  3--------0
            #  |        |
            #  |        |
            #  |________|
            #  2        1

                edges = [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                ]
                for edge in edges:
                    start_vertex = line[edge[0]]
                    end_vertex = line[edge[1]]
                    cv2.line(image, start_vertex, end_vertex, (255, 0, 255), 2)
                line = f.readline().strip()
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ =='__main__':
    get_2d_information()

