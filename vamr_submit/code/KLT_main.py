import matplotlib.pyplot as plt
from Initialization import *

from Update import *
from Dataset import *
from State import *
from collections import deque
import pdb

def draw_cloud(points3D:np.ndarray)->np.ndarray:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(
            points3D[:, 0],
            points3D[:, 2],
            -points3D[:, 1],
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-30, 30])
        plt.ylim([0, 20])
        plt.savefig('point_cloud.jpg')
        plt.close()

def main():

#------------------------------------load data---------------------------------------------
    # dataset=Parking()
    dataset=Kitti()
    # dataset=Malaga()
    # dataset=Datahome()

    K=dataset.K
    poses=dataset.poses
    img_num=dataset.image_number

    initial_images = [0, 5]

    img=State(dataset.get_image(initial_images[0]))
    img.set_id(initial_images[0])

    img1=State(dataset.get_image(initial_images[1]))
    img1.set_id(initial_images[1])


# -----------------------Initialization-----------------------------------------------------------

    state_history = deque(maxlen=20)


    sift = cv2.SIFT_create(1000,5)
    kp, des = sift.detectAndCompute(img.image_gray, None)
    kp1,des1= sift.detectAndCompute(img1.image_gray,None)

    kp=cv2.KeyPoint_convert(kp)
    kp1=cv2.KeyPoint_convert(kp1)

    matcher=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)

    # match_cv=matcher.knnMatch(des,des1,k=2)
    match_cv=matcher.match(des,des1)
    match_cv = sorted(match_cv, key=lambda x: x.distance)
    matches=convert_match(match_cv[:300])

    #select the matched keypoints for following steps
    kp_matched=kp[matches[:,0]]
    kp1_matched=kp1[matches[:,1]]

    img.set_kps(kp_matched)
    img1.set_kps(kp1_matched)

    #8-points algorithm to initialize the pose
    E,_=cv2.findEssentialMat(kp_matched,kp1_matched,K,cv2.RANSAC,0.999,1)
    _, R1, t1, _ = cv2.recoverPose(E, kp_matched, kp1_matched, K)

    #recoverPose can not obtain the t scale.

    if dataset.id==0 or dataset.id==1:
        pose_1=np.array([poses[initial_images[1],3],poses[initial_images[1],7],poses[initial_images[1],11]])
        pose_0 = np.array([poses[initial_images[0], 3], poses[initial_images[0], 7], poses[initial_images[0], 11]])
        scale=np.linalg.norm(pose_1-pose_0)/np.linalg.norm(t1)
    else:
        scale=1

    img.set_pose(np.eye(3), np.zeros((3, 1)))
    img1.set_pose(R1,scale*t1)

    #Triangulate to get initial 3D points
    points3D,img_kps,img1_kps=klt_triangulate(K,img,img1,5)
    img.set_kps(img_kps)
    img.set_3d_points(points3D)
    img1.set_kps(img1_kps)
    img1.set_3d_points(points3D)

    state_history.appendleft(img)
    state_history.appendleft(img1)

#-----------------------Update---------------------------------------------------------------

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.6,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(21, 21),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.05))


    harris_patch_size = 5
    harris_kappa = 0.01
    query_keypoint_num = 3
    nonmaximum_supression_radius = 5

    threshold=dataset.threshold

    img_pre=img

    candidate_mask=np.full(kp.shape,True,dtype=bool)
    candidate_mask[matches[:,0],:]=False

    harris=Harris(harris_patch_size, harris_kappa, query_keypoint_num, nonmaximum_supression_radius)
    img_pre.candidate_kps=get_new_candidate(img,feature_params,img_kps,harris)
    img_pre.first_obs=img_pre.candidate_kps
    img_pre.first_t=np.repeat(np.expand_dims(img_pre.t,axis=0),img_pre.candidate_kps.shape[0],axis=0)
    img_pre.first_R=np.repeat(np.expand_dims(img_pre.R,axis=0),img_pre.candidate_kps.shape[0],axis=0)

    # t is the trajectory
    t=np.zeros((1,3))
    # points is the 3D points to show
    points=np.zeros((0,3))

    plt.ion()
    plt.show()
    for i in range(initial_images[0],img_num):
        if i in initial_images:
            continue

        print("------------------------------------------------------------------------")
        print("current_image",i)
        print("reference",img_pre.id)

        img_cur=State(dataset.get_image(i))
        img_cur.set_id(i)

        kps_cur, _, _ = cv2.calcOpticalFlowPyrLK(img_pre.image_gray, img_cur.image_gray, img_pre.kps, None, **lk_params)
        judge=np.logical_and(kps_cur[:,0]<img_cur.image_gray.shape[1],kps_cur[:,1]<img_cur.image_gray.shape[0])
        judge=np.logical_and(judge,kps_cur[:,1]>0)
        judge=np.logical_and(judge,kps_cur[:,0]>0)

        img_cur.kps=kps_cur[judge].astype((np.float32))
        img_cur.points3D=img_pre.points3D[judge].astype((np.float32))

        input_points3D=img_cur.points3D
        input_kps=img_cur.kps

        print("solvePnp input points num",input_points3D.shape)
        input_num=input_points3D.shape[0]
        # pdb.set_trace()

        _, R_cur, t_cur, inlier = cv2.solvePnPRansac(input_points3D,input_kps, K,None,
                                                     reprojectionError=8,
                                                     # flags=cv2.SOLVEPNP_EPNP,
                                                     # flags=cv2.SOLVEPNP_UPNP,
                                                     flags=cv2.SOLVEPNP_ITERATIVE,
                                                     iterationsCount=2000,
                                                     )
        # convert to the matrix
        R_cur, _ = cv2.Rodrigues(R_cur)

        if inlier is None:
            continue
        inlier_num=len(inlier)


        print("inlier num",inlier.shape)

        inlier=list(np.squeeze(inlier,axis=1))

        img_cur.set_pose(R_cur,t_cur)
        img_cur.set_kps(img_cur.kps[inlier])

        img_cur.set_3d_points(img_cur.points3D[inlier])


        candidate_kps_cur, _, _ = cv2.calcOpticalFlowPyrLK(img_pre.image_gray,
                                                           img_cur.image_gray,
                                                           img_pre.candidate_kps.astype(np.float32),
                                                           None, **lk_params)

        judge1 = np.logical_and(candidate_kps_cur[:, 0]>0,candidate_kps_cur[:,1]>0)
        judge1=np.logical_and(judge1,candidate_kps_cur[:,0]<img_cur.image_gray.shape[1])
        judge1=np.logical_and(judge1,candidate_kps_cur[:,1]<img_cur.image_gray.shape[0])
        #
        candidate_kps_cur=candidate_kps_cur[judge1]
        candidate_first_obs=img_pre.first_obs[judge1]
        first_t=img_pre.first_t[judge1,:]
        first_R=img_pre.first_R[judge1,:,:]
        #
        new_candidate,new_first_obs,new_first_t,new_first_R,new_points3D,new_kps=klt_triangulate_candidate(K,candidate_kps_cur,t_cur,R_cur,candidate_first_obs,first_t,first_R,threshold)
        print("new candidate",new_candidate.shape)
        #
        added_candidate=get_new_candidate(img_cur,feature_params,new_candidate,harris)
        print("added candidate",added_candidate.shape)
        added_first_t=np.repeat(np.expand_dims(img_cur.t,axis=0),added_candidate.shape[0],axis=0)
        added_first_R=np.repeat(np.expand_dims(img_cur.R,axis=0),added_candidate.shape[0],axis=0)
        added_first_observe=added_candidate
        #
        img_cur.candidate_kps=np.vstack((new_candidate,added_candidate))
        img_cur.first_t=np.concatenate((new_first_t,added_first_t),axis=0)
        img_cur.first_R=np.concatenate((new_first_R,added_first_R),axis=0)
        img_cur.first_obs=np.vstack((new_first_obs,added_first_observe))

        tmp=-img_cur.R.T@img_cur.t

        print(tmp)
        t=np.vstack((t,tmp.T))
        state_history.appendleft(img_cur)

        ax_image = plt.subplot2grid((2,4),(0,0),rowspan=1,colspan=2)
        ax_landmarknumber = plt.subplot2grid((2,4),(1,0),rowspan=1,colspan=1)
        ax_fulltraj = plt.subplot2grid((2,4),(1,1),rowspan=1,colspan=1)
        ax_localtraj = plt.subplot2grid((2,4),(0,2),rowspan=2,colspan=2)

        ax_image.imshow(img_cur.image)
        ax_image.set_title("Current image")

        ax_image.scatter(img_cur.kps[:, 0],
                    img_cur.kps[:, 1],
                    marker='x', c='r')


        for id in range(img_cur.kps.shape[0]):
            point_from = img_pre.kps[judge][inlier][id, :]
            point_to = img_cur.kps[id, :]
            y_values = [point_from[1], point_to[1]]
            x_values = [point_from[0], point_to[0]]

            ax_image.plot(x_values, y_values, 'g', linewidth=1)

        img_cur.kps=np.vstack((img_cur.kps,new_kps)).astype(np.float32)
        img_cur.points3D=np.vstack((img_cur.points3D,new_points3D))
        #
        ax_image.scatter(candidate_kps_cur[:,0],
                    candidate_kps_cur[:,1],
                    marker='x',c="y")

        for id in range(candidate_kps_cur.shape[0]):
            point_from = img_pre.candidate_kps[judge1][id, :]
            point_to = candidate_kps_cur[id, :]
            y_values = [point_from[1], point_to[1]]
            x_values = [point_from[0], point_to[0]]

            ax_image.plot(x_values, y_values, 'w', linewidth=1)


        if dataset.id==0:
            ax_fulltraj.plot(poses[initial_images[0]:i, 3], poses[initial_images[0]:i, 11], 'g', label="ground truth")
            ax_fulltraj.plot(t[:, 0], t[:, 2], label='trajectory')
            x_max = max(np.max(t[:, 0]),np.max(poses[initial_images[0]:i, 3]))
            x_min = min(np.min(t[:, 0]),np.min(poses[initial_images[0]:i, 3]))
            y_max = max(np.max(t[:, 2]),np.max(poses[initial_images[0]:i, 11]))
            y_min = min(np.min(t[:, 2]),np.min(poses[initial_images[0]:i, 11]))

        elif dataset.id==1:

            ax_fulltraj.plot(poses[initial_images[0]:i,3], poses[initial_images[0]:i, 11], 'g',label="ground truth")
            ax_fulltraj.plot(t[:,0],t[:,2],label='trajectory')
            x_max = max(np.max(t[:, 0]),np.max(poses[initial_images[0]:i, 3]))
            x_min = min(np.min(t[:, 0]),np.min(poses[initial_images[0]:i, 3]))
            y_max = max(np.max(t[:, 2]),np.max(poses[initial_images[0]:i, 11]))
            y_min = min(np.min(t[:, 2]),np.min(poses[initial_images[0]:i, 11]))

        elif dataset.id==2 or dataset.id==3:
            ax_fulltraj.plot(t[:, 0], t[:, 2], label='trajectory')
            x_max = np.max(t[:, 0])
            x_min = np.min(t[:, 0])
            y_max = np.max(t[:, 2])
            y_min = np.min(t[:, 2])
        
        axis_max = max(x_max,y_max)
        axis_min = min(x_min,y_min)
        ax_fulltraj.set_xlim([x_min-(axis_max-axis_min-x_max+x_min)/2,x_max+(axis_max-axis_min-x_max+x_min)/2])
        ax_fulltraj.set_ylim([y_min-(axis_max-axis_min-y_max+y_min)/2,y_max+(axis_max-axis_min-y_max+y_min)/2])
        ax_fulltraj.legend()
        ax_fulltraj.set_title("Full trajectory")

        landmark_num=[]
        relative_frame_id=range(0,-len(list(state_history)),-1)
        for state in state_history:
            landmark_num.append(state.points3D.shape[0])
        ax_landmarknumber.plot(relative_frame_id,landmark_num,'-',color = 'black')
        ax_landmarknumber.set_title("# tracked landmarks over last 20 frames")

        ax_localtraj.scatter(state_history[0].points3D[:,0],
                    state_history[0].points3D[:,2],
                    marker='.',c="black", label='landmark')
        ax_localtraj.plot(t[-20:, 0], t[-20:, 2], linewidth=3, label='trajectory')
        ax_localtraj.legend()
        ax_localtraj.set_title("Trajectory of last 20 frames and landmarks")


        img_pre=img_cur
        plt.pause(0.01)
        plt.clf()



if __name__ =="__main__":
    main()