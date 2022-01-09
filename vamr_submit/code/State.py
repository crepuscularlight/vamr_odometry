import cv2

class State:
    def __init__(self,image_dir):
        self.image=cv2.imread(image_dir)
        self.image_gray=cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)

        self.kps=None
        self.points3D=None
        self.candidate_kps=None
        self.first_obs=None
        self.first_R=None
        self.first_t=None

    def set_id(self,id):
        self.id=id

    def set_pose(self,R,t):
        self.R=R
        self.t=t

    def set_kps(self,kps):
        self.kps=kps

    def set_3d_points(self,points3D):
        self.points3D=points3D

