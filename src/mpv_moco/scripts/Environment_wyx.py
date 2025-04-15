import numpy as np
from enum import Enum


class Env_Type(Enum):
    Levelground = 0
    Upstair = 1
    Downstair = 2
    Upslope = 3
    Downslope = 4
    Obstacle = 5
    Unknown = 6


class Environment:
    def __init__(self) -> None:
        self.classification_model = None
        self.type_pred_from_nn = Env_Type.Levelground
        self.type_pred_buffer = np.zeros(10,dtype=np.uint64)
        self.pcd_2d = np.zeros([0,2])
        self.pcd_thin = np.zeros([0,2])
        self.img_binary = np.zeros((100,100)).astype('uint8')

    def pcd_to_binary_image(self, pcd):
        pass

    def pcd_thin(self):
        pass

    def classification_from_img(self):
        pass

            