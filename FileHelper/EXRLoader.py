import OpenEXR
import Imath
import array
import numpy as np


class CameraAttr:
    Position = np.array([0., 0., 0.])
    Rotation = np.array([0., 0., 0.])
    f = np.array([0., 0.])  # 真实环境中存在fx与fy不同的情况，统一形式方便后期计算

    def __init__(self, pos: np.array, rot: np.array, f: np.array):
        self.Position = pos
        self.Rotation = rot
        self.f = f


class OpenEXRLoader:
    def __init__(self):
        self.dtype = Imath.PixelType(Imath.PixelType.FLOAT)

    def windowSize(self, fileHeader):
        dw = fileHeader['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        return size

    # 只有一个通道，即是深度
    def parseEXRSingle(self, datapath):
        file = OpenEXR.InputFile(datapath)
        Z_line = np.array(array.array('f', file.channel('V', self.dtype)))
        size = self.windowSize(file.header())
        sizeWidth, sizeHeight = size

        return Z_line.reshape((sizeHeight, sizeWidth)).transpose(1, 0), (sizeWidth, sizeHeight)

    def parseEXRMulti(self, datapath):
        file = OpenEXR.InputFile(datapath)
        Z_line = np.array(array.array('f', file.channel('ViewLayer.Depth.Z', self.dtype)))
        size = self.windowSize(file.header())
        return Z_line.reshape(size), size

    def __call__(self, datapath, camerapath) -> np.array:
        datapath = str(datapath)
        camerapath = str(camerapath)
        depth, size = self.parseEXRSingle(datapath)
        hardware = self.parseHardware(camerapath)
        for file in hardware:
            file[2][1], file[2][2] = size
            file[2][1], file[2][2] = file[2][1] / 2, file[2][2] / 2
        depth = depth[np.newaxis, :]
        depth = depth.transpose(0, 2, 1)
        return depth, hardware

    # 所有距离单位均为mm，所有角度单位均为°
    # 返回值：以数组的形式返回，每个返回的格式为：
    # T:[x,y,z]空间坐标
    # R:[x,y,z]旋转角
    # f:[焦距，行，列]
    def parseHardware(self, filename='cameras.txt'):
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        cameraList = []
        for i in range(int(len(lines) / 4)):
            pos = np.fromstring(lines[i * 4 + 1], dtype=np.float32, sep=',').reshape((1, 3))
            rot = np.fromstring(lines[i * 4 + 2], dtype=np.float32, sep=',').reshape((1, 3))
            f = np.fromstring(lines[i * 4 + 3], dtype=np.float32, sep=',')
            f = np.pad(f, (0, 2), 'constant').reshape((1, 3))
            cam = np.concatenate([pos, rot, f], axis=0)
            cameraList += [cam]

        return cameraList