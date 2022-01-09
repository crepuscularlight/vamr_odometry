from load_data import *
import os


class Dataset:
    def __init__(self,dataset):
        self.name=dataset
        self.data_dir="../data/"+self.name+"/"

        self.image_number=0
        self.K=None
        self.poses=None


class Parking(Dataset):
    def __init__(self):
        super().__init__("parking")
        self.K=load_K(self.data_dir)
        self.poses=np.loadtxt(self.data_dir+"poses.txt")
        self.image_number=599
        self.id=0
        self.threshold=12

    def get_image(self,id):
        return self.data_dir+ f"images/img_"+f"{id}".zfill(5)+".png"


class Kitti(Dataset):
    def __init__(self):
        super().__init__("kitti")
        self.image_number=2760

        self.poses=np.loadtxt(self.data_dir+"poses/05.txt")
        self.times=np.loadtxt(self.data_dir+"05/times.txt")
        self.K=np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                        [0, 7.188560000000e+02, 1.852157000000e+02],
                        [0 ,0 ,1]])
        self.id=1
        self.threshold=0.25

    def get_image(self,id):
        if id<2761:
            return self.data_dir+"05/image_0/"+f"{id}".zfill(6)+".png"
        else:
            return self.data_dir + "05/image_1/"+ f"{id-2760}".zfill(6) + ".png"


class Malaga(Dataset):
    def __init__(self):
        super().__init__("malaga-urban-dataset-extract-07")

        self.image_number=4242
        self.poses=None
        self.K=np.array( [[621.18428, 0, 404.0076],
                          [0, 621.18428, 309.05989],
                          [0, 0, 1]])
        self.id=2
        self.threshold=0.25

    def get_image(self,id):
        a=os.listdir(self.data_dir+"malaga-urban-dataset-extract-07_rectified_800x600_Images")
        a=sorted(a)

        return self.data_dir+"malaga-urban-dataset-extract-07_rectified_800x600_Images/"+a[2*id]


class Datahome(Dataset):
    def __init__(self):
        super().__init__("datahome")

        self.poses=None
        self.K=np.array([[967.7941,0,520.9090],
                         [0,967.4235,346.8391],
                         [0,0,1]])
        self.image_number=270
        self.id=3
        self.threshold=1

    def get_image(self,id):
        a = os.listdir(self.data_dir+"JPEG")
        a=sorted(a)
        return self.data_dir+"JPEG/"+a[id]

