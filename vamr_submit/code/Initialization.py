import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

#Harris response

class Harris:

    def __init__(self,harris_patch_size,harris_kappa,
                 query_keypoint_num,
                 nonmaximum_supression_radius):

        self.harris_patch_size=harris_patch_size
        self.harris_kappa=harris_kappa
        self.kp_num=query_keypoint_num
        self.suppresion_radius=nonmaximum_supression_radius



    def calculate_Harris(self,img):
        # I_x=cv2.Sobel(self.img,ddepth=cv2.CV_8U,dx=1,dy=0,ksize=3)
        # I_y=cv2.Sobel(self.img,ddepth=cv2.CV_8U,dx=0,dy=1,ksize=3)
        I_x = convolve(img.astype('float'), [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        I_y = convolve(img.astype('float'), [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        I_x_2 = np.square(I_x)
        I_y_2 = np.square(I_y)

        I_x_mul_I_y = I_x * I_y

        sigma_I_x_2 = convolve(I_x_2, np.ones(shape=(self.harris_patch_size, self.harris_patch_size)))
        sigma_I_y_2 = convolve(I_y_2, np.ones(shape=(self.harris_patch_size, self.harris_patch_size)))
        sigma_I_x_mul_I_y = convolve(I_x_mul_I_y, np.ones(shape=((self.harris_patch_size, self.harris_patch_size))))

        det = sigma_I_x_2 * sigma_I_y_2 - np.square(sigma_I_x_mul_I_y)
        trace = sigma_I_x_2 + sigma_I_y_2

        tmp = det - self.harris_kappa * np.square(trace)
        tmp[tmp < 0] = 0
        self.Harris = tmp



    def select_keypoints(self,img):
        self.calculate_Harris(img)
        keypoint_coord = np.zeros((self.kp_num, 2))
        # scores=np.ones(shape=Harris.shape,dtype=float)
        # np.copyto(scores,Harris)
        scores = np.copy(self.Harris)
        scores = np.pad(scores, self.suppresion_radius)

        h, w = self.Harris.shape
        nonmax_radius=self.suppresion_radius


        for i in range(0, self.kp_num):
            max_index = np.argmax(scores)

            max_h, max_w = np.unravel_index(max_index, (h + 2 * nonmax_radius, w + 2 * nonmax_radius))
            keypoint_coord[i, :] = [max_w - nonmax_radius, max_h - nonmax_radius]
            scores[max_h - nonmax_radius:max_h + nonmax_radius, max_w - nonmax_radius:max_w + nonmax_radius] = 0

        keypoint_coord = keypoint_coord.astype(int)

        return keypoint_coord

    def distribute_keypoints(self,img):
        h,w=img.shape
        split_h=8
        split_w=8

        h_list=np.linspace(0,h,split_h+1)[:-1].astype(int)
        w_list=np.linspace(0,w,split_w+1)[:-1].astype(int)

        delta_h=int(h/split_h)
        delta_w=int(w/split_w)


        keypoint=np.zeros((0,2))
        for i in range(split_h):
            for j in range(split_w):
                keypoint_sub=self.select_keypoints(img[h_list[i]:h_list[i]+delta_h,w_list[j]:w_list[j]+delta_w])\
                             +np.array([[w_list[j],h_list[i]]])
                keypoint=np.vstack((keypoint,keypoint_sub))

        return keypoint



def klt_triangulate(K,im1,im2,threshold):
    kps_1=im1.kps

    kps_2=im2.kps

    R1=im1.R
    t1=im1.t

    R2=im2.R
    t2=im2.t

    if(len(t1.shape)<2):
        t1=np.expand_dims(t1, 1)
    if(len(t2.shape)<2):
        t2=np.expand_dims(t2, 1)



    proj_mat1=np.concatenate((R1,t1),axis=1)
    proj_mat2=np.concatenate((R2,t2),axis=1)

    points3D = cv2.triangulatePoints(K@proj_mat1,
                                     K@proj_mat2,
                                     kps_1.T,
                                     kps_2.T
                                     )

    points3D=points3D[:3]/points3D[3]
    points3D=points3D.T

    # # We need to keep track of the correspondences between image points and 3D points
    #
    # # TODO
    # # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
    # # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
    points3D_homo=np.append(points3D,np.ones((points3D.shape[0],1)),axis=1)

    points3D_1_camera=(np.append(R1, t1, 1)@points3D_homo.T).T
    points3D_2_camera = (np.append(R2, t2, 1) @ points3D_homo.T).T

    judge1=np.logical_and(points3D_1_camera[:,2]>0,points3D_2_camera[:,2]>0)

    points3D =points3D[judge1]
    im1_kps=kps_1[judge1]
    im2_kps=kps_2[judge1]

    return points3D,im1_kps,im2_kps



def convert_match(match_cv):
    matches=np.zeros((0,2))


    for match in match_cv:
        # match=match[0]
        matches=np.vstack((matches,np.array([[match.queryIdx,match.trainIdx]])))
    matches=matches.astype(int)


    return matches

if __name__ =="__main__":
  print(np.__version__)