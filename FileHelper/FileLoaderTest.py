from FileLoadHelper import SynDatasetLoader


loader = SynDatasetLoader()
depth , left , right , rightF , rightMV , rightRT = loader.LoadData()
print(depth)
print(left)
print(right)
print(rightF)
print(rightRT)
print(rightMV)
# loader = SourceFileLoader("/data/jsj/dev/cpp/DisparityToCut/build-Debug/202457test")
# preprocess = Preprocess()
# for i in range(loader.SumLen):
#     pcd , imgL , imgR = loader.LoadData()
#     preprocess.align(pcd , imgL)