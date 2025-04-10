import numpy as np
import math
import datetime
from sklearn import linear_model, datasets
from scipy.signal import buttap, lp2hp_zpk, bilinear_zpk, zpk2tf, butter


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
    
    def get_fea(self, _print_=False, ax=None, idx=0, nn_class_prior=0):
        t0 = datetime.datetime.now()
        self.env_type = nn_class_prior
        if nn_class_prior == 1 or nn_class_prior == 2 or nn_class_prior == 5: #重新识别类型
            self.env_type = self.get_env_type_up_or_down(ax, nn_class_prior)
        if _print_:
            print("Env_Type:{}".format(self.env_type))
        if self.env_type == 0 or self.env_type == 3 or self.env_type == 4:
            self.get_fea_slope(_print_, ax, idx)
        elif self.env_type == 1:
            self.get_fea_sa(_print_,ax, idx)
        elif self.env_type == 2:
            self.get_fea_sd(_print_,ax, idx)
        elif self.env_type == 5:
            self.get_fea_ob(_print_, ax, idx)

    def get_fea_slope(self, _print_=False, ax=None, idx=0):
        pass

    def get_env_type_up_or_down(self, ax=None, nn_class_prior=0):
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
        
        elif line1_success and not line2_success:
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1


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

        if line1_success and not line2_success:
            self.stair_low_x, self.stair_low_y, self.stair_low_idx = x1, y1, idx1
            self.stair_high_x, self.stair_high_y, self.stair_high_idx = x1, y1, idx1


    def get_fea_ob(self, _print_=False, ax=None, idx=0):
        pass
    
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
    
    def rebudacy_check_up_left_sa(self, stair_low_x, stair_low_y, ax=None):
        pass