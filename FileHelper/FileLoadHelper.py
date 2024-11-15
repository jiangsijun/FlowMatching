import sys

import torch

sys.path.append('FileHelper')

from PicLoader import OpenCVPicLoader
from EXRLoader import OpenEXRLoader
from PLYLoader import O3dPlyLoader

from pathlib import Path 


#目录层级：
# \dataset\场景名（多个）\view1\
# 包括的图像名称
class SynDatasetLoader():
    def __init__(self , rootPath:str = './dataset/') -> None :
        self.rootPath = Path(rootPath)
        assert(self.rootPath.exists() and self.rootPath.is_dir())

        self.sourceList = []
        self.formDataList()
        self.SumLen = len(self.sourceList)
        self.depthLoader = OpenEXRLoader()
        self.picLoader = OpenCVPicLoader()
        self.iter = 0
        pass


    def formDataList(self):
        scenepath = sorted(self.rootPath.glob('*'))
        for scene in scenepath :
            viewspath = sorted(scene.glob('*'))
            print(viewspath)
            for view in viewspath :
                if not view.joinpath('ImageDepth0001.exr').exists():
                    print(view," path not exist" )
                    continue
                # depthpath = view.joinpath('ImageDepth0001.exr')
                # leftImage = view.joinpath('ImageLeft0001.png')
                # rightImage = view.joinpath('ImageRight0001.png')
                # rightF = view.joinpath('ImageRightf0001.png')
                # rightMV = view.joinpath('ImageRightMV0001.png')
                # rightRT = view.joinpath('ImageRightRT0001.png')
                # type2
                depthpath = view.joinpath('ImageDepth0001.exr')
                leftImage = view.joinpath('Image0001.png')
                rightImage = view.joinpath('Image0002.png')
                rightF = view.joinpath('Image0006.png')
                rightMV = view.joinpath('Image0004.png')
                rightRT = view.joinpath('Image0008.png')
                # type3
                # depthpath = view.joinpath('ImageDepth0001.exr')
                # leftImage = view.joinpath('0001.png')
                # rightImage = view.joinpath('0002.png')
                # rightF = view.joinpath('0003.png')
                # rightMV = view.joinpath('0005.png')
                # rightRT = view.joinpath('0004.png')
                cameras = view.joinpath('cameras.txt')
                self.sourceList.append({"depth":depthpath.absolute(),"left":leftImage.absolute(),"right":rightImage.absolute(),
                                        "rightF":rightF.absolute(),"rightMV": rightMV.absolute() ,
                                        "rightRT": rightRT.absolute(),"cameras":cameras.absolute()})

        return

    def LoadMatrix(self , matrixPath='checkMatrix.txt'):
        assert self.rootPath.joinpath(matrixPath).exists()
        
        #先不写
        

        pass

    def LoadData(self):
        item = self.sourceList[self.iter]
        depthpic , cam = self.depthLoader(item['depth'],item['cameras'])
        leftpic = self.picLoader(item['left'])
        rightpic = self.picLoader(item['right'])
        rightFPic = self.picLoader(item['rightF'])
        rightMVPic = self.picLoader(item['rightMV'])
        rightRTPic = self.picLoader(item['rightRT'])
        self.iter = self.iter +1 
        return depthpic , leftpic , rightpic , rightFPic , rightMVPic , rightRTPic ,cam
        
class LidarDatasetLoader():
    #打开文件夹
    def __init__(self, rootPath:str):
        self.rootPath = Path(rootPath)
        self.sourceList = [] #里面挂的是切片，[pcd,_1.jpg,_2.jpg]
        assert(self.checkRootPath())
        self.formDataList()
        self.SumLen = len(self.sourceList)
        self.iter = 0

    def checkRootPath(self):
        if self.rootPath.exists() and self.rootPath.is_dir(): 
            return True
        return False

    #读取Matrix.txt
    def LoadMatrix(self , matrixPath='checkMatrix.txt'):
        assert self.rootPath.joinpath(matrixPath).exists()
        
        #先不写
        

        pass

    def formDataList(self):
        point_files = sorted(self.rootPath.glob('*.pcd'))
        for file in point_files:
            fixfile = file.stem 
            jpg1 = self.rootPath.joinpath(fixfile+'_1.jpg')
            jpg2 = self.rootPath.joinpath(fixfile+'_2.jpg')
            if jpg1.exists() and jpg2.exists():
                self.sourceList.append([file.absolute(),jpg1.absolute(),jpg2.absolute()]) 
        return        
        
    #读取点云与左右图，并以链表的形式返回？
    def LoadData(self):
        assert(self.iter < self.SumLen)
        pcdPath , imgLPath , imgRPath = self.sourceList[self.iter]
        pcd = o3d.io.read_point_cloud(pcdPath.as_posix())
        imgL = cv2.imread(imgLPath.as_posix())
        imgR = cv2.imread(imgRPath.as_posix())
        self.iter+=1
        return pcd , imgL , imgR


    

    

