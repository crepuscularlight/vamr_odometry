import numpy as np
import cv2
from Initialization import *


def reproject(p_W_corners,K,R_prime,t):

    R_t=np.concatenate((R_prime,t),axis=1)

    p_W_corners_1=np.append(p_W_corners,np.ones((p_W_corners.shape[0],1)),axis=1)

    uv1=K@R_t@(p_W_corners_1.T)

    uv_2d,lamda=uv1[0:2,:],uv1[2,:]
    uv=np.divide(uv_2d,lamda)
    uv=uv.T
    return uv



def klt_triangulate_candidate(K,candidate_kps_cur,t_cur,R_cur,first_obs,first_t,first_R,threshold):
    candidate_num=candidate_kps_cur.shape[0]
    proj_mat2 = np.concatenate((R_cur, t_cur), axis=1)

    new_candidate=np.zeros((0,2))
    new_first_obs=np.zeros((0,2))
    new_first_t=np.zeros((0,3,1))
    new_first_R=np.zeros((0,3,3))

    new_points3D=np.zeros((0,3))
    new_kps=np.zeros((0,2))
    for i in range(candidate_num):
        R1=first_R[i]
        t1=first_t[i]

        proj_mat1 = np.concatenate((R1, t1), axis=1)

        point3D=cv2.triangulatePoints(K@proj_mat1,
                                      K@proj_mat2,
                                      first_obs[i],
                                      candidate_kps_cur[i])
        point3D=point3D/point3D[3]

        point3D_1_camera = proj_mat1@point3D
        point3D_2_camera = proj_mat2@point3D

        point3D=point3D[:3]
        point3D_o1=point3D+R1.T@t1
        point3D_o2=point3D+R_cur.T@t_cur

        cosine=np.sum(point3D_o1*point3D_o2)/(np.linalg.norm(point3D_o1)*np.linalg.norm(point3D_o2))

        arccos=np.arccos(cosine)*180/np.pi

        if point3D_1_camera[2] > 0 and point3D_2_camera[2] > 0 and arccos>threshold:
            new_points3D=np.vstack((new_points3D,point3D.T))
            new_kps=np.vstack((new_kps,candidate_kps_cur[i]))
        else:
            new_candidate=np.vstack((new_candidate,candidate_kps_cur[i]))
            new_first_t=np.concatenate((new_first_t,np.expand_dims(t1,axis=0)),axis=0)
            new_first_R=np.concatenate((new_first_R,np.expand_dims(R1,axis=0)),axis=0)
            new_first_obs=np.vstack((new_first_obs,first_obs[i]))


    return new_candidate,new_first_obs,new_first_t,new_first_R,new_points3D,new_kps




def get_new_candidate(img_cur,feature_params,new_candidate,harris):
    # calculate_Harris()
    # added_kp = cv2.goodFeaturesToTrack(img_cur.image_gray, mask=None, **feature_params)
    # added_kp = np.array(added_kp)
    # added_kp = np.squeeze(added_kp, axis=1)
    # added_kp,_=sift.detectAndCompute(img_cur.image_gray,None)
    # added_kp=cv2.KeyPoint_convert(added_kp)


    added_kp=harris.distribute_keypoints(img_cur.image_gray)

    candidate_mask=np.full(shape=img_cur.image_gray.shape,fill_value=False,dtype=bool).T

    candidate_mask[list(added_kp.T.astype(int))]=True
    candidate_mask[list(new_candidate.T.astype(int))]=False

    tmp=np.where(candidate_mask)

    added_candidate=np.array(np.where(candidate_mask)).T
    added_candidate=added_candidate.astype(np.float32)

    return added_candidate



