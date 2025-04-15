import numpy as np
import math
import datetime
from sklearn import linear_model, datasets
from scipy.signal import buttap, lp2hp_zpk, bilinear_zpk, zpk2tf, butter
from scipy.ndimage import gaussian_filter1d
from pcd_fast_plot import FastPlotCanvas

def RANSAC(X, y, th):
    def is_data_valid(X_subset, y_subset):
        x = X_subset
        y = y_subset

        if abs(x[1] - x[0]) < 0.025:
            return False
        else:
            k = (y[1] - y[0]) / (x[1] - x[0])

        theta = math.atan(k)

        if abs(theta) < th:
            r = True
        else:
            r = False

        return r

    ransac = linear_model.RANSACRegressor(min_samples=2, residual_threshold=0.03,
                                          is_data_valid=is_data_valid,
                                          max_trials=500)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    # Predict calibrate of estimated models
    line_X = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    theta_line = math.atan((line_y_ransac[-1] - line_y_ransac[0]) / (line_X[-1] - line_X[0]))
    return inlier_mask, outlier_mask, line_y_ransac, line_X, theta_line


class pcd_operator_system(object):
    def __init__(self, pcd_new):
        self.Acenter = np.zeros((0, 2))
        self.Bcenter = np.zeros((0, 2))
        self.Ccenter = np.zeros((0, 2))
        self.fea_A = np.zeros((0, 2))
        self.fea_B = np.zeros((0, 2))
        self.fea_C = np.zeros((0, 2))
        self.fea_D = np.zeros((0, 2))
        self.fea_E = np.zeros((0, 2))
        self.fea_F = np.zeros((0, 2))
        self.is_fea_A_gotten = False  # 是否提取到A
        self.is_fea_B_gotten = False  # 是否提取到B
        self.is_fea_C_gotten = False  # 是否提取到C
        self.is_fea_D_gotten = False
        self.is_fea_E_gotten = False
        self.is_fea_F_gotten = False
        self.corner_situation = 0
        self.pcd_new = pcd_new
        self.num_line = 0
        self.fea_extra_over = False
        self.env_type = 0
        self.env_rotate = 0
        self.cost_time = 0
        self.slope = 0
        self.w_stair, self.h_stair = 0, 0 
        self.d_obs = 0
    
    def get_fea(self, _print_=False, ax=None, idx=0, nn_class_prior=0):
        t0 = datetime.datetime.now()
        self.env_type = nn_class_prior
        if nn_class_prior == 1 or nn_class_prior == 2 or nn_class_prior == 5: #重新识别类型
            self.env_type = self.get_env_type_stair_and_obstacle(ax, nn_class_prior)
        if _print_:
            print("Env_Type:{}".format(self.env_type))
        if self.env_type == 0 or self.env_type == 3 or self.env_type == 4:
            self.get_fea_slope(_print_, ax, idx)
            self.fea_extra_over = True
        elif self.env_type == 1:
            self.get_fea_sa(_print_,ax, idx)
            self.fea_extra_over = True
        elif self.env_type == 2:
            self.get_fea_sd(_print_,ax, idx)
            self.fea_extra_over = True
        elif self.env_type == 5:
            self.get_fea_ob(_print_, ax, idx)
            self.fea_extra_over = True

    def get_fea_slope(self, _print_=False, ax=None, idx=0):
        pass

    def get_env_type_stair_and_obstacle(self, ax=None, nn_class_prior=0):
        idx_xmin, idx_xmax = np.argmin(self.pcd_new[:,0]), np.argmax(self.pcd_new[:,0])
        xmin, xmax = self.pcd_new[idx_xmax,0], self.pcd_new[idx_xmin,1]
        y_xmin, y_xmax = self.pcd_new[idx_xmax,1], self.pcd_new[idx_xmin,1]
        
        # 最高一级平面    
        line_highest = self.pcd_new[np.where(np.abs(self.pcd_new[:, 1] - np.max(self.pcd_new[:,1])) < 0.015)[0], :]
        y_line_hest = np.nanmean(line_highest[:,1])
    
        xmin_line_hest = line_highest[np.argmin(line_highest[:,0]),0]
        xmax_line_hest = line_highest[np.argmax(line_highest[:,0]),0]
        
        # 判断有没有点云在最高一级平面的左侧或者右侧
        idx_right_part = np.where(np.logical_and(self.pcd_new[:, 0] > xmax_line_hest + 0.02,
                                                 self.pcd_new[:, 1] < y_line_hest - 0.03))[0]
        idx_left_part = np.where(np.logical_and(self.pcd_new[:, 0] < xmin_line_hest - 0.02,
                                                self.pcd_new[:, 1] < y_line_hest - 0.03))[0]
        check_right = False if np.shape(idx_right_part[0])==0 else True
        check_left = False if np.shape(idx_left_part[0])==0 else True

        if not check_left and not check_right:
            env_type = 0 #重新矫正为平地
            return env_type
        elif not check_left and check_right:
            # 有点云在右边，可能为下楼
            ymin_right = np.min(self.pcd_new[idx_right_part, 1])
            line_lowest = self.pcd_new[np.where(np.abs(self.pcd_new[:, 1] - ymin_right) < 0.015)[0], :]
            y_line_lest = np.nanmean(line_lowest[:, 1])
            x_line_lest = line_lowest[np.argmax(line_lowest[:, 0]), 0]
            if y_line_hest-y_line_hest>0.05 and x_line_lest-xmax_line_hest > 0.04:
                env_type = 2
                return env_type
            else:
                env_type = 0
                return env_type
        elif not check_right and check_left:
            # 有点云在左边，可能为上楼
            ymin_left = np.min(self.pcd_new[idx_left_part, 1])
            line_lowest = self.pcd_new[np.where(np.abs(self.pcd_new[:, 1] - ymin_right) < 0.015)[0], :]
            y_line_lest = np.nanmean(line_lowest[:, 1])
            x_line_lest = line_lowest[np.argmin(line_lowest[:, 0]), 0]
            if y_line_hest-y_line_hest>0.05 and xmin_line_hest-x_line_lest > 0.04:
                env_type = 1
                return env_type
            else:
                env_type = 0
                return env_type
        elif check_left and check_right:
            # 有点云在右边和左边，可能为障碍
            line_lowest_right = self.pcd_new[idx_right_part,:]
            ymin_right = np.min(line_lowest_right[:, 1])
            line_lowest_right = line_lowest_right[np.where(np.abs(line_lowest_right[:, 1] - ymin_right) < 0.015)[0], :]
            y_right_lest = np.nanmean(line_lowest_right[:, 1])
            x_right_lest = line_lowest_right[np.argmax(line_lowest_right[:, 0]), 0]
            ###
            line_lowest_left = self.pcd_new[idx_left_part, :]
            ymin_left = np.min(line_lowest_left[:, 1])
            line_lowest_left = line_lowest_left[np.where(np.abs(line_lowest_left[:, 1] - ymin_left) < 0.015)[0], :]
            y_left_lest = np.nanmean(line_lowest_left[:, 1])
            x_left_lest = line_lowest_left[np.argmin(line_lowest_left[:, 0]), 0]

            if y_line_hest - y_left_lest > 0.03 and y_line_hest - y_right_lest > 0.03:
                if xmin_line_hest - x_left_lest > 0.02 and x_right_lest - xmax_line_hest > 0.02:
                    env_type = 5
                    self.obs_high_level_x = line_highest[:,0]
                    self.obs_high_mean_y = y_line_hest
                    self.obs_low_x = xmin_line_hest
                    self.obs_low_y = y_left_lest
                    return env_type
            elif abs(y_line_hest - y_left_lest) < 0.03 and y_line_hest-ymin_left > 0.08:
                if xmin_line_hest - x_left_lest > 0.1:
                    env_type = 1
                    return env_type
            elif abs(y_line_hest - y_left_lest) < 0.03 and y_line_hest - ymin_right > 0.08:
                if x_right_lest - xmax_line_hest > 0.1:
                    env_type = 2
                    return env_type
            elif abs(y_line_hest - y_left_lest) < 0.03 and abs(y_line_hest - y_right_lest) < 0.03:
                env_type = 0
                return env_type
        return 0
    
    def get_fea_sa(self, _print_=False, ax=None, idx=0):
        line1_success = False
        x1, y1, idx1 = [], [], []
        mean_line1 = 0
        x1, y1, mean_line1, idx1, line1_success = self.ransac_process_1(th_ransac_k=0.15, th_length=0.05,
                                                                        th_interval=0.15, _print_=_print_)
        if line1_success:
            self.num_line = 1
        else:
            self.num_line = 0
            return

        line2_success = False
        x2, y2, idx2 = [], [], []
        mean_line2 = 0
        if line1_success:
            x2, y2, mean_line2, idx2, line2_success = self.ransac_process_2(idx1, th_ransac_k=0.12, th_length=0.02,
                                                                            th_interval=0.1, _print_=_print_)
            if line2_success:
                self.num_line = 2
            else:
                self.num_line = 1
        self.need_to_check_B = False
        self.need_to_check_A = False
        self.need_to_check_C = False

        if line1_success and line2_success:
            if mean_line1 > mean_line2:
                # line1 is higher than line2
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x2, y2, idx2
            else:
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x2, y2, idx2
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.rebudacy_check_up_right_sa(self.stair_high_x, self.stair_high_y,ax)
            self.rebundacy_check_under_left_sa(self.stair_low_x, self.stair_low_y,ax)
            self.classify_corner_situation_sa(num_stair=2, _print_=_print_)
            self.get_fea_A(_print_)
            self.get_fea_B(_print_)
            self.get_fea_C(_print_)

        if line1_success and not line2_success:
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
            self.rebudacy_check_up_right_sa(self.stair_high_x, self.stair_high_y,ax)
            self.rebundacy_check_under_left_sa(self.stair_low_x, self.stair_low_y,ax)
            self.classify_corner_situation_sa(num_stair=1, _print_=_print_)
            self.get_fea_A(_print_)
            self.get_fea_B(_print_)
            self.get_fea_C(_print_)

    def get_fea_sd(self, _print_=False, ax=None, idx=0):
        line1_success = False
        x1, y1, idx1 = [], [], []
        mean_line1 = 0
        x1, y1, mean_line1, idx1, line1_success = self.ransac_process_1(th_ransac_k=0.15,th_interval=0.1,
                                                                        th_length=0.1,_print_=_print_)
        if line1_success:
            self.num_line = 1
        else:
            self.num_line = 0 
            return
        # idx1_new = np.where(self.pcd_new[:,1]>mean_line1-0.02)[0]
        
        line2_success = False
        x2, y2, idx2 = [], [], []
        mean_line2 = 0
        if line1_success:
            x2, y2, mean_line2, idx2, line2_success = self.ransac_process_2(idx1, th_ransac_k=0.12, th_length=0.1,
                                                                            th_interval=0.05, _print_=_print_)
            if line2_success:
                self.num_line = 2
            else:
                self.num_line = 1
        self.need_to_check_D = False
        self.need_to_check_E = False
        self.need_to_check_F = False

        if line1_success and line2_success:
            if mean_line1 > mean_line2:
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x2, y2, idx2
            else:
                self.stair_high_x, self.stair_high_y, self.stair_high_idx = x2, y2, idx2
                self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.rebundacy_check_under_right_sd(self.stair_low_x, self.stair_low_y, ax)
            self.rebundacy_check_up_left_sd(self.stair_high_x, self.stair_high_y, ax)
            self.classify_corner_situation_sd(num_stair=2, _print_=_print_)
            self.get_fea_D(_print_)
            self.get_fea_E(_print_)
            self.get_fea_F(_print_)
        
        elif line1_success and not line2_success:
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.rebundacy_check_under_right_sd(self.stair_low_x, self.stair_low_y, ax)
            self.rebundacy_check_up_left_sd(self.stair_high_x, self.stair_high_y, ax)
            self.classify_corner_situation_sd(num_stair=2, _print_=_print_)
            self.get_fea_D(_print_)
            self.get_fea_E(_print_)
            self.get_fea_F(_print_)

    def get_fea_ob(self, _print_=False, ax=None, idx=0):
        self.corner_situation = 8
        self.need_to_check_B = True
        self.need_to_check_C = True
        self.need_to_check_D = True
        Bcenter_x = self.obs_low_x
        Bcenter_y = self.obs_low_y
        idx_fea_B = np.logical_and(np.abs(self.pcd_new[:, 0] - Bcenter_x) < 0.01,
                                   np.abs(self.pcd_new[:, 1] - Bcenter_y) < 0.01)
        if np.shape(self.pcd_new[idx_fea_B, 0])[0] > 10:
            fea_Bx_new = self.pcd_new[idx_fea_B, 0].reshape((-1, 1))
            fea_By_new = self.pcd_new[idx_fea_B, 1].reshape((-1, 1))
            if _print_:
                print("find feature B:{},{}".format(Bcenter_x, Bcenter_y))
        else:
            mean_Bx = Bcenter_x
            mean_By = Bcenter_y
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Bx_new = mean_Bx + rand * 0.001
            rand = np.random.rand(20).reshape((-1, 1))
            fea_By_new = mean_By + rand * 0.001
            if _print_:
                print("complete feature B:{},{}".format(Bcenter_x, Bcenter_y))
        self.Bcenter = np.array([Bcenter_x, Bcenter_y])
        self.fea_B = np.hstack([fea_Bx_new, fea_By_new])
        self.is_fea_B_gotten = True
        ###############
        Ccenter_x = np.min(self.obs_high_level_x)
        Ccenter_y = self.obs_high_mean_y
        idx_fea_C = np.logical_and(np.abs(self.pcd_new[:, 0] - Ccenter_x) < 0.02,
                                   np.abs(self.pcd_new[:, 1] - Ccenter_y) < 0.02)

        if np.shape(self.pcd_new[idx_fea_C, 0])[0] > 10:
            fea_Cx_new = self.pcd_new[idx_fea_C, 0].reshape((-1, 1))
            fea_Cy_new = self.pcd_new[idx_fea_C, 1].reshape((-1, 1))
            if _print_:
                print("find feature C:{},{}".format(Ccenter_x, Ccenter_y))
        else:
            mean_Cx = Ccenter_x
            mean_Cy = Ccenter_y
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Cx_new = mean_Cx + rand * 0.001
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Cy_new = mean_Cy + rand * 0.001
            if _print_:
                print("complete feature C:{},{}".format(Ccenter_x, Ccenter_y))
        self.Ccenter = np.array([Ccenter_x, Ccenter_y])
        self.fea_C = np.hstack([fea_Cx_new, fea_Cy_new])
        self.is_fea_C_gotten = True
        #################
        Dcenter_x = np.max(self.obs_high_level_x)
        Dcenter_y = self.obs_high_mean_y
        idx_fea_D = np.logical_and(np.abs(self.pcd_new[:, 0] - Dcenter_x) < 0.02,
                                   np.abs(self.pcd_new[:, 1] - Dcenter_y) < 0.02)
        if np.shape(self.pcd_new[idx_fea_D, 0])[0] > 10:
            fea_Dx_new = self.pcd_new[idx_fea_D, 0].reshape((-1, 1))
            fea_Dy_new = self.pcd_new[idx_fea_D, 1].reshape((-1, 1))
            if _print_:
                print("find feature D:{},{}".format(Dcenter_x, Dcenter_y))
        else:
            mean_Dx = Dcenter_x
            mean_Dy = Dcenter_y
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Dx_new = mean_Dx + rand * 0.001
            rand = np.random.rand(20).reshape((-1, 1))
            fea_Dy_new = mean_Dy + rand * 0.001
            if _print_:
                print("complete feature C:{},{}".format(Dcenter_x, Dcenter_y))
        self.Dcenter = np.array([Dcenter_x, Dcenter_y])
        self.fea_D = np.hstack([fea_Dx_new, fea_Dy_new])
        self.is_fea_D_gotten = True
            
    def ransac_process_1(self, th_ransac_k=0.1, th_length=0.1, th_interval=0.05, _print_=False):
        x1, y1 = [], []
        mean_line1 = 0
        idx1 = []
        line1_success = False
        # 初始点云
        X0 = self.pcd_new[:,0].reshape((-1,1))
        Y0 = self.pcd_new[:,1].reshape((-1,1))


        def ransac_process(X,Y,th_ransac_k):
            inlier_mask, outlier_mask, line_y_ransac, line_X, theta_line = RANSAC(X,Y,th_ransac_k)
            mean_line = np.nanmean(Y[inlier_mask])#line1的均值
            idx_x_in_X = np.where(abs(Y-mean_line)<0.015)[0]#将靠近line1均值的点统一纳入line1
            x, y = X[idx_x_in_X,:], Y[idx_x_in_X,:]
            return mean_line, idx_x_in_X, x, y

        
        idx_X0_in_X0 = np.arange(np.shape(X0)[0])
        idx_x1_in_X0 = np.zeros((0,))
        try:
            mean_line1, idx_x1_in_X0, x1, y1 = ransac_process(X0, Y0, th_ransac_k)
            line1_length = np.max(x1)-np.min(x1)
            diff_x1 = np.diff(x1, axis=0)
            # 条件：
            # 1.直线点数>20
            # 2.直线长度>th_length
            # 3.点和点之间水平距离<th_interval
            if np.shape(idx_x1_in_X0)[0]>20 and line1_length>=th_length and np.max(np.abs(diff_x1))<th_interval:
                line1_success = True
                if _print_:
                    print("Line1 get")
            elif np.shape(idx_x1_in_X0)[0] > 20 and line1_length < th_length and np.max(np.abs(diff_x1)) < th_interval:
                # 直线过短，重新提取一次
                if _print_:
                    print("Line1 Try RANSAC Again")
                X_temp = np.delete(X0, idx_x1_in_X0).reshape((-1, 1))
                Y_temp = np.delete(Y0, idx_x1_in_X0).reshape((-1, 1))
                idx_Xtemp_in_X0 = np.delete(idx_X0_in_X0, idx_x1_in_X0)
                
                mean_line1, idx_x1_in_Xtemp, _, _ = ransac_process(X_temp, Y_temp, th_ransac_k)
                idx_x1_in_X0 = idx_Xtemp_in_X0[idx_x1_in_Xtemp]
                x1, y1 = X0[idx_x1_in_X0,:], Y0[idx_x1_in_X0,:]
                line1_length = np.max(x1) - np.min(x1)
                diff_x1 = np.diff(x1, axis=0)
                # 条件：
                # 1.直线点数>20
                # 2.直线长度>th_length
                # 3.点和点之间水平距离<th_interval
                if np.shape(idx_x1_in_X0)[0] > 20 and line1_length > th_length and np.max(np.abs(diff_x1)) < th_interval:
                    line1_success = True
                    if _print_:
                        print("Line1 get")
                        something_print = 1
                else:
                    line1_success = False
                    if _print_:
                        print("Not get Line1")
                        print("line1_length:{}<{}".format(line1_length, th_length) + 
                              "or diff_x1:{}>{}".format(np.max(np.abs(diff_x1)), th_interval))
        except Exception as e:
            line1_success = False
            if _print_:
                print(e)
                print("Line1 RANSAC False, No Line in the picture")
        return x1, y1, mean_line1, idx_x1_in_X0, line1_success
    
    def ransac_process_2(self, idx_x1_in_X0, th_ransac_k=0.1, th_length=0.1, th_interval=0.05, _print_=False):
        x2, y2 = [], []
        mean_line2 = 0
        idx2 = []
        line2_success = False
        X0 = self.pcd_new[:, 0].reshape((-1, 1))
        Y0 = self.pcd_new[:, 1].reshape((-1, 1))
        idx_X0_in_X0 = np.arange(np.shape(X0)[0])
        X1 = np.delete(X0, idx_x1_in_X0).reshape((-1,1))
        Y1 = np.delete(Y0, idx_x1_in_X0).reshape((-1,1))
        idx_X1_in_X0 = np.delete(idx_X0_in_X0, idx_x1_in_X0)
        
        idx_x2_in_X0 = np.zeros((0,))
        def ransac_process(X,Y,th_ransac_k):
            inlier_mask, outlier_mask, line_y_ransac, line_X, theta_line = RANSAC(X,Y,th_ransac_k)
            mean_line = np.nanmean(Y[inlier_mask])#line1的均值
            idx_x_in_X = np.where(abs(Y-mean_line)<0.015)[0]#将靠近line1均值的点统一纳入line1
            x, y = X[idx_x_in_X,:], Y[idx_x_in_X,:]
            return mean_line, idx_x_in_X, x, y

        try:
            mean_line2, idx_x2_in_X1, x2, y2 = ransac_process(X1, Y1, th_ransac_k)
            idx_x2_in_X0 = idx_X1_in_X0[idx_x2_in_X1]
            x2, y2 = X0[idx_x2_in_X0,:], Y0[idx_x2_in_X0,:]
            line2_length = np.max(x2)-np.min(x2)
            diff_x2 = np.diff(x2, axis=0)
            if np.shape(idx_x2_in_X0)[0] > 20 and line2_length >= th_length and np.max(abs(diff_x2)) < th_interval:
                line2_success = True
                if _print_:
                    print("Line2 get")
            elif np.shape(idx_x2_in_X0)[0] > 20 and np.max(abs(diff_x2)) < th_interval and line2_length < th_length:
                if _print_:
                    print("Line2 RANSAC Try again")
                X_temp = np.delete(X1, idx_x2_in_X1).reshape((-1, 1))
                Y_temp = np.delete(Y1, idx_x2_in_X1).reshape((-1, 1))
                idx_Xtemp_in_X1 = np.delete(idx_X1_in_X0, idx_x2_in_X1)
                idx_Xtemp_in_X0 = idx_X1_in_X0[idx_Xtemp_in_X1]
                mean_line2, idx_x2_in_Xtemp,_,_ = ransac_process(X_temp, Y_temp, th_ransac_k)
                idx_x2_in_X0 = idx_Xtemp_in_X0[idx_x2_in_Xtemp]
                x2, y2 = X0[idx_x2_in_X0,:], Y0[idx_x2_in_X0,:]
                line2_length = np.max(x2)-np.min(x2)
                diff_x2 = np.diff(x2, axis=0)
                if np.shape(idx_x2_in_X0)[0] > 20 and line2_length >= th_length and np.max(abs(diff_x2)) < th_interval:
                    line2_success = True
                    if _print_:
                        print("Line2 get")
                else:
                    line2_success = False
                    if _print_:
                        print("Not get Line2")
                        print(
                            "line2_length:{}<{}".format(line2_length, th_length) + "diff_x2:{}>{}".format(
                                np.max(np.abs(diff_x2)), th_interval))
        except Exception as e:
            line2_success = False
            if _print_:
                print(e)
                print("Line2 RANSAC False")
        return x2, y2, mean_line2, idx_x2_in_X0, line2_success
    
    def rebudacy_check_up_right_sa(self, stair_high_x, stair_high_y, ax=None):
        ymax, xmax = np.max(self.pcd_new[:,1]), np.max(self.pcd_new[:,0])
        # 检查是否存在最高级台阶上方的垂直立面
        self.has_part_up_line = False
        if ymax - np.nanmean(stair_high_y) > 0.01:
            idx_up = np.logical_and(
                self.pcd_new[:,1]>np.nanmean(stair_high_y),
                np.abs(self.pcd_new[:,0]-np.max(stair_high_x))<0.015
            )
            up_normal_part = self.pcd_new[idx_up,:]
            diff_up = np.diff(up_normal_part,axis=0)
            idx_continuous = np.where(np.abs(diff_up[:,1])<0.005)[0]
            up_continuous_part = up_normal_part[idx_continuous, :]
            if np.shape(idx_continuous)[0] > 10 and np.max(up_continuous_part[:, 1]) - np.nanmean(stair_high_y) > 0.005:
                self.has_part_up_line = True
        else:
            self.has_part_up_line = False
        # 检查是否存在最高级台阶上方的一级平面
        self.has_part_right_line = False
        if self.has_part_up_line and xmax - np.max(stair_high_x) > 0.02:
            right_max_y = self.pcd_new[np.argmax(self.pcd_new[:, 0]), 1]
            if right_max_y - np.nanmean(stair_high_y) > 0.01:
                idx_up_right = np.where(np.abs(self.pcd_new[:, 1] - right_max_y) < 0.015)[0]
                up_right_part = self.pcd_new[idx_up_right, :]
                diff_up_right = np.diff(up_right_part, axis=0)
                idx_continuous = np.where(np.abs(diff_up_right[:, 0]) < 0.005)[0]
                right_part_continuous = up_right_part[idx_continuous]
                if np.shape(right_part_continuous)[0] > 10 and np.max(right_part_continuous[:,0]) - np.max(stair_high_x) > 0.04:
                    self.has_part_right_line = True
                    self.pcd_right_up_part_sa = right_part_continuous
        else:
            self.has_part_right_line = False
        
    def rebundacy_check_under_left_sa(self, stair_low_x, stair_low_y, ax=None):
        ymin, xmin = np.min(self.pcd_new[:,1]), np.min(self.pcd_new[:,0])
        # 检查是否存在最低台阶下方的垂直面
        self.has_part_under_line = False
        if np.nanmean(stair_low_y) - ymin > 0.02:
            idx_under =  np.logical_and(self.pcd_new[:, 1] < np.nanmean(stair_low_y),
                                    np.abs(self.pcd_new[:, 0] - np.min(stair_low_x)) < 0.015)
            under_normal_part = self.pcd_new[idx_under,:]
            diff_under = np.diff(under_normal_part, axis=0)
            idx_continuous = np.where(np.abs(diff_under[:,1])<0.005)[0]
            under_continuous_part = under_normal_part[idx_continuous,:]
            if np.shape(idx_continuous)[0] > 10 and np.nanmean(stair_low_y) - np.min(under_continuous_part[:, 1]) > 0.005:
                self.has_part_under_line = True
        else:
            self.has_part_under_line = False
        # 检查是否存在最低台阶下方的平面
        self.has_part_left_line = False
        if self.has_part_under_line and np.min(stair_low_x) - xmin > 0.02:
            left_min_y = self.pcd_new[np.argmin(self.pcd_new[:, 0]), 1]
            if np.nanmean(stair_low_y)-left_min_y > 0.01:
                idx_under_left = np.where(np.abs(self.pcd_new[:, 1] - left_min_y) < 0.015)[0]
                under_left_part = self.pcd_new[idx_under_left, :]
                diff_under_left = np.diff(under_left_part, axis=0)
                idx_continuous = np.where(np.abs(diff_under_left[:, 0]) < 0.005)[0]
                left_part_continuous = under_left_part[idx_continuous]
                if np.shape(left_part_continuous)[0] > 10 and np.min(stair_low_x) - np.min(left_part_continuous) > 0.04:
                    self.has_part_left_line = True
        else:
            self.has_part_left_line = False
    
    def classify_corner_situation_sa(self, num_stair, _print_=False):
        def set_flags(situation, A, B, C, disable_right=False):
            self.corner_situation = situation
            self.need_to_check_A = A
            self.need_to_check_B = B
            self.need_to_check_C = C
            if disable_right:
                self.has_part_right_line = False

        if num_stair == 2:
            if self.has_part_under_line:
                if self.has_part_up_line:
                    set_flags(3, True, True, True, disable_right=True)
                elif not self.has_part_up_line:
                    set_flags(3, True, True, True)
            elif not self.has_part_under_line:
                if self.has_part_up_line:
                    set_flags(4, False, True, True, disable_right=True)
                elif not self.has_part_up_line:
                    set_flags(4, False, True, True)
        else:
            if self.has_part_under_line and self.has_part_left_line:
                if self.has_part_up_line and self.has_part_right_line:
                    set_flags(3, True, True, True)
                elif self.has_part_up_line and not self.has_part_right_line:
                    set_flags(1, True, True, False)
                elif not self.has_part_up_line:
                    set_flags(5, False, False, True)
            elif self.has_part_under_line and not self.has_part_left_line:
                if self.has_part_up_line and self.has_part_right_line:
                    set_flags(3, True, True, True)
                elif self.has_part_up_line and not self.has_part_right_line:
                    set_flags(1, True, True, False)
                elif not self.has_part_up_line:
                    set_flags(5, False, False, True)
            elif not self.has_part_under_line:
                if self.has_part_up_line and self.has_part_right_line:
                    set_flags(4, False, True, True)
                elif self.has_part_up_line and not self.has_part_right_line:
                    set_flags(2, False, True, False)

        if _print_:
            print(f"corner_situation:{self.corner_situation}")

    def get_fea_A(self, _print_=False):
        if self.need_to_check_A:
            Acenter_x = np.min(self.stair_low_x)
            Acenter_y = self.stair_low_y[np.argmin(self.stair_low_x)][0]
            idx_fea_A = np.logical_and(np.abs(self.pcd_new[:, 0] - Acenter_x) < 0.01,
                                       np.abs(self.pcd_new[:, 1] - Acenter_y) < 0.01)
            if np.shape(self.pcd_new[idx_fea_A, 0])[0] > 10:
                fea_Ax_new = self.pcd_new[idx_fea_A, 0].reshape((-1, 1))
                fea_Ay_new = self.pcd_new[idx_fea_A, 1].reshape((-1, 1))
                if _print_:
                    print("find feature A:{},{}".format(Acenter_x, Acenter_y))
            else:
                mean_Ax = Acenter_x
                mean_Ay = Acenter_y
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Ax_new = mean_Ax + rand * 0.0001
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Ay_new = mean_Ay + rand * 0.0001
                if _print_:
                    print("complete feature A:{},{}".format(Acenter_x, Acenter_y))
            self.Acenter = np.array([Acenter_x, Acenter_y])
            self.fea_A = np.hstack([fea_Ax_new, fea_Ay_new])
            self.is_fea_A_gotten = True
        else:
            self.is_fea_A_gotten = False
    
    def get_fea_B(self, _print_=False):
        if self.need_to_check_B:
            Bcenter_x = max(np.min(self.stair_high_x), np.max(self.stair_low_x))
            Bcenter_y = np.nanmean(self.stair_low_y)
            idx_fea_B = np.logical_and(np.abs(self.pcd_new[:, 0] - Bcenter_x) < 0.01,
                                       np.abs(self.pcd_new[:, 1] - Bcenter_y) < 0.01)
            if np.shape(idx_fea_B)[0] > 10:
                fea_Bx_new = self.pcd_new[idx_fea_B, 0].reshape((-1, 1))
                fea_By_new = self.pcd_new[idx_fea_B, 1].reshape((-1, 1))
                if _print_:
                    print("find feature B:{},{}".format(Bcenter_x, Bcenter_y))
            else:
                mean_Bx = Bcenter_x
                mean_By = Bcenter_y
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Bx_new = mean_Bx + rand * 0.0001
                rand = np.random.rand(15).reshape((-1, 1))
                fea_By_new = mean_By + rand * 0.0001
                if _print_:
                    print("complete feature B:{},{}".format(Bcenter_x, Bcenter_y))
            self.Bcenter = np.array([Bcenter_x, Bcenter_y])
            self.fea_B = np.hstack([fea_Bx_new, fea_By_new])
            self.is_fea_B_gotten = True
        else:
            self.is_fea_B_gotten = False

    def get_fea_C(self, _print_=False):
        if self.need_to_check_C:
            if self.has_part_right_line:
                X_right_part = self.pcd_right_up_part_sa[:, 0]
                Y_right_part = self.pcd_right_up_part_sa[:, 1]
                Ccenter_x = np.min(X_right_part)
                Ccenter_y = Y_right_part[np.argmin(X_right_part)]
            else:
                Ccenter_x = np.min(self.stair_high_x)
                Ccenter_y = np.nanmean(self.stair_high_y)
            
            idx_fea_C = np.logical_and(np.abs(self.pcd_new[:, 0] - Ccenter_x) < 0.01,
                                       np.abs(self.pcd_new[:, 1] - Ccenter_y) < 0.01)
            if np.shape(idx_fea_C)[0] > 10:
                fea_Cx_new = self.pcd_new[idx_fea_C, 0].reshape((-1, 1))
                fea_Cy_new = self.pcd_new[idx_fea_C, 1].reshape((-1, 1))
                if _print_:
                    print("find feature C:{},{}".format(Ccenter_x, Ccenter_y))
            else:
                mean_Cx = Ccenter_x
                mean_Cy = Ccenter_y
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Cx_new = mean_Cx + rand * 0.0001
                rand = np.random.rand(15).reshape((-1, 1))
                fea_Cy_new = mean_Cy + rand * 0.0001
                if _print_:
                    print("complete feature C:{},{}".format(Ccenter_x, Ccenter_y))
            self.Ccenter = np.array([Ccenter_x, Ccenter_y])
            self.fea_C = np.hstack([fea_Cx_new, fea_Cy_new])
            self.is_fea_C_gotten = True
        else:
            self.is_fea_C_gotten = False
    
    def rebundacy_check_up_left_sd(self, stair_high_x, stair_high_y, ax=None):
        ymax = np.max(self.pcd_new[:,1])
        self.has_part_up_line = False
        if ymax - np.nanmean(stair_high_y) > 0.02:
            idx_up = np.logical_and(
                self.pcd_new[:,1] > np.nanmean(stair_high_y),
                np.abs(self.pcd_new[:,0]-np.min(stair_high_x))<0.015
            )
            up_normal_part = self.pcd_new[idx_up,:]
            diff_up = np.diff(up_normal_part, axis=0)
            idx_continuous = np.where(np.abs(diff_up[:,1])<0.005)[0]
            up_continuous_part = up_normal_part[idx_continuous, :]
            if np.shape(up_continuous_part)[0] > 10 and np.max(up_continuous_part[:, 1]) - np.nanmean(stair_high_y) > 0.005:
                self.has_part_up_line = True
            else:
                self.has_part_up_line = False
        self.has_part_left_line = False
        if self.has_part_up_line:
            self.has_part_left_line = True
            self.pcd_left_up_part_sd = up_continuous_part
    
    def rebundacy_check_under_right_sd(self, stair_low_x, stair_low_y, ax=None):
        ymin = np.min(self.pcd_new[:,1])
        self.has_part_under_line = False
        if np.nanmean(stair_low_y) - ymin > 0.02:
            idx_under =  np.logical_and(self.pcd_new[:, 1] < np.nanmean(stair_low_y),
                                    np.abs(self.pcd_new[:, 0] - np.max(stair_low_x)) < 0.015)
            under_normal_part = self.pcd_new[idx_under,:]
            diff_under = np.diff(under_normal_part, axis=0)
            idx_continuous = np.where(abs(diff_under[:, 0]) < 0.005)[0]
            under_continuous_part = under_normal_part[idx_continuous,:]
            if np.shape(idx_continuous)[0] > 10 and np.nanmean(stair_low_y) - np.min(under_continuous_part[:, 1]) > 0.005:
                self.has_part_under_line = True
            else:
                self.has_part_under_line = False
        self.has_part_right_line = False
        if self.has_part_under_line:
            self.has_part_right_line = True
        
    def classify_corner_situation_sd(self, num_stair, _print_=False):
        def set_flags(situation, D, E, F, disable_under=False):
            self.corner_situation = situation
            self.need_to_check_D = D
            self.need_to_check_E = E
            self.need_to_check_F = F
            if disable_under:
                self.has_part_under_line, self.has_part_right_line = False, False
        
        if num_stair == 2:
            if self.has_part_up_line and self.has_part_left_line:
                set_flags(10, True, True, True,disable_under=True)
            elif not self.has_part_up_line and not self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    set_flags(10,True, True, True,disable_under=False)
                elif not self.has_part_under_line and not self.has_part_right_line:
                    set_flags(9, True, True, False, disable_under=False)
        else:       
            if self.has_part_up_line and self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    set_flags(10,True,True,True,disable_under=False)
                elif not self.has_part_under_line and not self.has_part_right_line:
                    set_flags(9, True, True, False, disable_under=False)
            elif not self.has_part_up_line and not self.has_part_left_line:
                if self.has_part_under_line and self.has_part_right_line:
                    set_flags(9, True, True, False, disable_under=False)
                elif not self.has_part_under_line and not self.has_part_right_line:
                    set_flags(0, False, False, False, disable_under=False)

    def get_fea_D(self, _print_=False):
        if self.need_to_check_D:
            if self.has_part_up_line:
                Dcenter_x = np.max(self.pcd_left_up_part_sd[:, 0])
                Dcenter_y = self.pcd_left_up_part_sd[np.argmax(self.pcd_left_up_part_sd[:, 0]), 1]
            elif not self.has_part_up_line:
                Dcenter_x = np.max(self.stair_high_x[:, 0])
                Dcenter_y = self.stair_high_y[np.argmax(self.stair_high_x[:, 0])][0]
            idx_fea_D = np.logical_and(np.abs(self.pcd_new[:, 0] - Dcenter_x) < 0.02,
                                       np.abs(self.pcd_new[:, 1] - Dcenter_y) < 0.02)
            if np.shape(idx_fea_D)[0] > 10:
                    fea_Dx_new = self.pcd_new[idx_fea_D, 0].reshape((-1, 1))
                    fea_Dy_new = self.pcd_new[idx_fea_D, 1].reshape((-1, 1))
                    if _print_:
                        print("find feature C:{},{}".format(Dcenter_x, Dcenter_y))
            else:
                mean_Dx = Dcenter_x
                mean_Dy = Dcenter_y
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Dx_new = mean_Dx + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Dy_new = mean_Dy + rand * 0.001
                if _print_:
                    print("complete feature D:{},{}".format(Dcenter_x, Dcenter_y))
            self.Dcenter = np.array([Dcenter_x, Dcenter_y])
            self.fea_D = np.hstack([fea_Dx_new, fea_Dy_new])
            self.is_fea_D_gotten = True
        else:
            self.is_fea_D_gotten = False        

    def get_fea_E(self, _print_=False):
        if self.need_to_check_E:
            if self.has_part_up_line:
                Ecenter_x = min(np.min(self.stair_high_x), np.max(self.pcd_left_up_part_sd[:, 0]))
                Ecenter_y = np.nanmean(self.stair_high_y)
            else:
                Ecenter_x = min(np.max(self.stair_high_x), np.min(self.stair_low_x))
                Ecenter_y = np.nanmean(self.stair_low_y)

            idx_fea_E = np.logical_and(np.abs(self.pcd_new[:, 0] - Ecenter_x) < 0.02,
                                       np.abs(self.pcd_new[:, 1] - Ecenter_y) < 0.02)

            if np.shape(self.pcd_new[idx_fea_E, 0])[0] > 10:
                fea_Ex_new = self.pcd_new[idx_fea_E, 0].reshape((-1, 1))
                fea_Ey_new = self.pcd_new[idx_fea_E, 1].reshape((-1, 1))
                if _print_:
                    print("find feature E:{},{}".format(Ecenter_x, Ecenter_y))
                else:
                    mean_Ex = Ecenter_x
                    mean_Ey = Ecenter_y
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ex_new = mean_Ex + rand * 0.001
                    rand = np.random.rand(20).reshape((-1, 1))
                    fea_Ey_new = mean_Ey + rand * 0.001
                    if _print_:
                        print("complete feature E:{},{}".format(Ecenter_x, Ecenter_y))
            self.Ecenter = np.array([Ecenter_x, Ecenter_y])
            self.fea_E = np.hstack([fea_Ex_new, fea_Ey_new])
            self.is_fea_E_gotten = True
        else:
            self.is_fea_E_gotten = False

    def get_fea_F(self, _print_=False):
        if self.need_to_check_F:
            if self.has_part_up_line:
                Fcenter_x = np.max(self.stair_high_x)
                Fcenter_y = np.nanmean(self.stair_high_y)
            else:
                Fcenter_x = np.max(self.stair_low_x)
                Fcenter_y = np.nanmean(self.stair_low_y)
            idx_fea_F = np.logical_and(np.abs(self.pcd_new[:, 0] - Fcenter_x) < 0.02,
                                       np.abs(self.pcd_new[:, 1] - Fcenter_y) < 0.02)
            if np.shape(self.pcd_new[idx_fea_F, 0])[0] > 10:
                fea_Fx_new = self.pcd_new[idx_fea_F, 0].reshape((-1, 1))
                fea_Fy_new = self.pcd_new[idx_fea_F, 1].reshape((-1, 1))
                if _print_:
                    print("find feature B:{},{}".format(Fcenter_x, Fcenter_y))
            else:
                mean_Fx = Fcenter_x
                mean_Fy = Fcenter_y
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Fx_new = mean_Fx + rand * 0.001
                rand = np.random.rand(20).reshape((-1, 1))
                fea_Fy_new = mean_Fy + rand * 0.001
                if _print_:
                    print("complete feature F:{},{}".format(Fcenter_x, Fcenter_y))
            self.Fcenter = np.array([Fcenter_x, Fcenter_y])
            self.fea_F = np.hstack([fea_Fx_new, fea_Fy_new])
            self.is_fea_F_gotten = True
        else:
            self.is_fea_F_gotten = False
    
    def show_fast(self, fast_plot_ax:FastPlotCanvas, type, id=0, p_text=[-0.5,0], 
                  p_pcd=None, downsample=2):
        if p_pcd is None:
            p_pcd = [0, 0]
        fast_plot_ax.set_pcd(self.pcd_new[::downsample]+p_pcd)
        if self.fea_extra_over:
            if self.env_type == 1:
                if self.is_fea_A_gotten:
                    fast_plot_ax.set_fea_A(self.fea_A[0:-1:5] + p_pcd)
                if self.is_fea_B_gotten:
                    fast_plot_ax.set_fea_B(self.fea_B[0:-1:5] + p_pcd)
                if self.is_fea_C_gotten:
                    fast_plot_ax.set_fea_C(self.fea_C[0:-1:5] + p_pcd)
            elif self.env_type == 2:
                if self.is_fea_D_gotten:
                    fast_plot_ax.set_fea_D(self.fea_D[0:-1:5] + p_pcd)
                if self.is_fea_E_gotten:
                    fast_plot_ax.set_fea_E(self.fea_E[0:-1:5] + p_pcd)
                if self.is_fea_F_gotten:
                    fast_plot_ax.set_fea_F(self.fea_F[0:-1:5] + p_pcd)

            elif self.env_type == 5:
                if self.is_fea_B_gotten:
                    fast_plot_ax.set_fea_B(self.fea_B[0:-1:5] + p_pcd)
                if self.is_fea_C_gotten:
                    fast_plot_ax.set_fea_C(self.fea_C[0:-1:5] + p_pcd)
                if self.is_fea_D_gotten:
                    fast_plot_ax.set_fea_D(self.fea_D[0:-1:5] + p_pcd)
            fast_plot_ax.set_info(p_text[0], p_text[1], type, id, self.corner_situation, self.env_rotate)

    def fea_to_env_paras(self):
        if self.env_type == 0 or self.env_type == 3 or self.env_type == 4:
            x = self.pcd_new[::2,0]
            z = gaussian_filter1d(self.pcd_new[::2,1], 2)
            coefs = np.polyfit(x, z, deg=10)
            slope_rad = np.arctan(coefs[0])
            slope_deg = np.round(np.rad2deg(slope_rad),2)
            self.slope = slope_deg
        elif self.env_type == 1:
            if self.corner_situation == 3:
                self.w_stair = self.Bcenter[0]-self.Acenter[0]
                self.h_stair = self.Ccenter[1]-self.Bcenter[1]
            elif self.corner_situation == 1:
                self.w_stair = self.Bcenter[0]-self.Acenter[0]
            elif self.corner_situation == 4:
                self.h_stair = self.Ccenter[1]-self.Bcenter[1]
        elif self.env_type == 2:
            if self.corner_situation == 9:
                self.w_stair = self.Ecenter[0]-self.Dcenter[0]
            elif self.corner_situation == 10:
                self.w_stair = self.Ecenter[0]-self.Dcenter[0]
                self.h_stair = self.Dcenter[1]-self.Ecenter[1]
        elif self.env_type == 5:
            self.w_stair = self.Dcenter[0]-self.Ccenter[0]
            self.h_stair = self.Ccenter[1]-self.Bcenter[1]
            self.d_obs = self.Ccenter[0]