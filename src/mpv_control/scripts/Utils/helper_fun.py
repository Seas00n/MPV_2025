import lcm
from core.impedance_lcm.exlcm import impedance_info
from Utils.FSM_Para import *

def fifo_mat(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat

class ImpedanceUpdate:
    def __init__(self, k_mat, b_mat, q_e_mat):
        self.q_e_mat_new = q_e_mat
        self.k_mat_new = k_mat
        self.b_mat_new = b_mat
        self.impedance_need_to_update = False 
        self.impedance_msg = impedance_info()
        self.v_est = 0.6
        self.s_est = 0
    def impedance_callback(self, channel, data):
        msg = self.impedance_msg.decode(data)
        para_list = [msg.para_phase1, 
                 msg.para_phase2,
                 msg.para_phase3,
                 msg.para_phase4]
        self.v_est = msg.velocity_estimated 
        self.s_est = msg.slope_estimated
        for phase_i in range(4):
            self.k_mat_new[idx_mode_level_ground][idx_knee][phase_i] = para_list[phase_i][0]
            self.b_mat_new[idx_mode_level_ground][idx_knee][phase_i] = para_list[phase_i][1]
            self.q_e_mat_new[idx_mode_level_ground][idx_knee][phase_i] = para_list[phase_i][2]
            self.k_mat_new[idx_mode_level_ground][idx_ankle][phase_i] = para_list[phase_i][3]
            self.b_mat_new[idx_mode_level_ground][idx_ankle][phase_i] = para_list[phase_i][4]
            self.q_e_mat_new[idx_mode_level_ground][idx_ankle][phase_i] = para_list[phase_i][5]
        print("New Impedance Parameters get")
        print(self.k_mat_new[idx_mode_level_ground])
        print(self.b_mat_new[idx_mode_level_ground])
        print(self.q_e_mat_new[idx_mode_level_ground])
        self.impedance_need_to_update = True