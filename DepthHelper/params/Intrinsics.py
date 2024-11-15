import numpy as np

class IntrinsicsParam:
    m_fx = 0.0
    m_fy = 0.0
    m_cx = 0.0
    m_cy = 0.0
    m_matrix = np.zeros((3,3),dtype=float)

    def __init__(self, fx , fy , cx , cy ) :
        self.m_fx = fx
        self.m_fy = fy
        self.m_cx = cx 
        self.m_cy = cy
        self.m_matrix[0,0] = fx
        self.m_matrix[0,2] = cx
        self.m_matrix[1,1] = fy
        self.m_matrix[1,2] = cy
        self.m_matrix[2,2] = 1.

    def SetParam(self, fx, fy, cx, cy):
        self.m_fx = fx
        self.m_fy = fy
        self.m_cx = cx 
        self.m_cy = cy
        self.m_matrix[0,0] = fx
        self.m_matrix[0,2] = cx
        self.m_matrix[1,1] = fy
        self.m_matrix[1,2] = cy
        self.m_matrix[2,2] = 1.

    def SetMatrix(self , inner:np.array):
        self.m_fx = inner[0,0]
        self.m_cx = inner[0,2]
        self.m_fy = inner[1,1]
        self.m_cy = inner[1,2]
        self.m_matrix = inner


    def GetParam(self):
        return self.m_fx , self.m_fy, self.m_cx, self.m_cy
    
    def GetMatrix(self):
        return self.m_matrix
    
