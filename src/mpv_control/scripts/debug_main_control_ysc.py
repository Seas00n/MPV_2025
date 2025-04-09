import numpy as np
import time
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
mpv_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.append(mpv_dir+"/mpv_moco/scripts/")
from Utils.FSM_Para import *
import Utils.Motor as Motor
from Utils.helper_fun import *
import select
import threading



q_e_mat = np.array(
    [[[10,  8, 70, 20], [ 5, 12, -11, 1]],
     [[ 6,  6, 75, 55], [ 2, 10, -11, 0]], 
     [[10, 40, 70, 10], [-5,  5, -5, -10]],
     [[5, 30, 70, 5], [5, 15, -8, 2]], 
     [[5, 25, 70, 2], [5, 15, -8, 2]]]).astype(np.float16)
k_mat = np.array([[[100, 100, 20, 15], [75, 90, 30, 30]], 
                  [[60, 60, 16, 14], [50, 80, 20, 20]],
                  [[40, 80, 19, 18], [50, 50, 20, 20]], 
                  [[40, 40, 14, 8], [50, 50, 20, 20]],
                  [[40, 40, 14, 8], [50, 50, 20, 20]]]).astype(np.float16)
b_mat = np.array([[[5, 10, 1.05, 2.05], [5, 5, 1, 1]], 
                  [[5, 5, 1, 1], [6, 6, 1, 1]], 
                  [[5, 5, 1, 1], [6, 6, 1, 1]],
                  [[5, 5, 1, 1], [6, 6, 1, 1]], 
                  [[5, 5, 1, 1], [6, 6, 1, 1]]]).astype(np.float16)

My_push = [20,10,10,22,22]
Fz_touch = [15,25,25,15,15]
Fz_swing = 220/120*140

data_example = np.load("./data/q_example.npy")

pros_state_bf = np.memmap(pros_state_buffer_path, mode='r+', dtype='float32', shape=(9,))

imp_update = ImpedanceUpdate(k_mat=k_mat, b_mat=b_mat, q_e_mat=q_e_mat)


def update_phase(phase, My_min, Fz_max, Fz_min, q_k_buf, q_a_buf, idx_env_mode=idx_mode_level_ground):
    global k_mat, b_mat, q_e_mat
    global Fz_swing, My_push
    global imp_update

    if phase == 0:
        if My_min > My_push[idx_env_mode]:
            phase = 1
            print('enter push-off stage', My_min)
        else:
            phase = 0
    elif phase == 1:
        if Fz_max < Fz_swing:
            phase = 2
            print('enter flexion stage', Fz_max)
        else:
            phase = 1
    elif phase == 2:
        if np.min(q_k_buf) > q_e_mat[idx_env_mode][idx_knee][idx_swing_flexion]-10:
            phase = 3
            print('enter extension stage')
            if imp_update.impedance_need_to_update:
                q_e_mat = imp_update.q_e_mat_new
                k_mat = imp_update.k_mat_new
                b_mat = imp_update.b_mat_new
                # set_threshold_swing_with_task(imp_update.v_est, imp_update.s_est)
                imp_update.impedance_need_to_update = False
        else:
            phase = 2
    elif phase == 3:
        if Fz_min > Fz_touch[idx_env_mode]:
            phase = 0
            print('enter early stance stage')
        else:
            phase = 3        
    return phase


def update_impedance_paras(idx_phase, idx_env_mode = idx_mode_level_ground):
    desired_k = [k_mat[idx_env_mode][idx_knee][idx_phase],
                k_mat[idx_env_mode][idx_ankle][idx_phase]]
    desired_b = [b_mat[idx_env_mode][idx_knee][idx_phase],
                b_mat[idx_env_mode][idx_ankle][idx_phase]]
    desired_qe = [q_e_mat[idx_env_mode][idx_knee][idx_phase],
                q_e_mat[idx_env_mode][idx_ankle][idx_phase]]
    
    # Km=0.11 ic=6.1 n=50
    desired_k[0] /= 1000/(50*0.11*6.1)
    desired_k[1] /= 1000/(50*0.11*6.1)
    desired_b[0] /= 1000/(50*0.11*6.1)
    desired_b[1] /= 1000/(50*0.11*6.1)

    return desired_k, desired_b, desired_qe

def cal_stance_control_command(desired_k, desired_b, desired_qe):
    q_qv_d = np.zeros((8,))
    q_qv_d[0], q_qv_d[2] = desired_qe[0], desired_qe[1]
    q_qv_d[1], q_qv_d[3] = 0, 0
    q_qv_d[4], q_qv_d[6] = desired_k[0], desired_k[1]
    q_qv_d[5], q_qv_d[7] = desired_b[0], desired_b[1]
    return q_qv_d

def lcm_impedance_job():
    time.sleep(2)
    lc = lcm.LCM()
    impedance_sub = lc.subscribe("Impedance_Info",imp_update.impedance_callback)
    try:
        while True:
            rfds, wfds, efds = select.select([lc.fileno()], [], [], 0.02)
            if rfds:
                lc.handle()
    except KeyboardInterrupt:
        pass
    lc.unsubscribe(impedance_sub)


if __name__ == "__main__":
    
    six_force_buf = np.memmap(six_force_buffer_path, dtype='float32', 
                          mode='r',shape=(6,))

    six_force_smooth_buffer = np.zeros((5,6))
    q_knee_smooth_buffer = np.zeros((10,))
    q_ankle_smooth_buffer = np.zeros((10,))

    for i in range(np.shape(six_force_smooth_buffer)[0]):
        six_force_once = np.copy(six_force_buf[:])
        
        time.sleep(0.05)
    
    impedance_lcm_job = threading.Thread(target=lcm_impedance_job)
    impedance_lcm_job.start()

    t0 = time.time()
    phase = 0
    debug_ptr = 0
    try:
        while True:
            tn = time.time()-t0
            q_qv, i_t = np.zeros((4,)), np.zeros((4,))
            q_qv[0] = data_example[debug_ptr,4]-data_example[debug_ptr,3]
            q_qv[1] = data_example[debug_ptr,4]+data_example[debug_ptr,5]

            six_force_once = np.copy(six_force_buf[:])
            
            q_knee_smooth_buffer = fifo_mat(q_knee_smooth_buffer,q_qv[0])
            q_ankle_smooth_buffer = fifo_mat(q_ankle_smooth_buffer,q_qv[1])
            six_force_smooth_buffer = fifo_mat(six_force_smooth_buffer,six_force_once)
            Fz_max = np.max(six_force_smooth_buffer[:,2])
            Fz_min = np.min(six_force_smooth_buffer[:,2])
            My_min = np.min(six_force_smooth_buffer[:,4])
            
            phase = int(data_example[debug_ptr, -1])
            if phase == 3:
                if imp_update.impedance_need_to_update:
                    q_e_mat = imp_update.q_e_mat_new
                    k_mat = imp_update.k_mat_new
                    b_mat = imp_update.b_mat_new
                    # set_threshold_swing_with_task(imp_update.v_est, imp_update.s_est)
                    imp_update.impedance_need_to_update = False
            
            desired_k, desired_b, desired_qe = update_impedance_paras(phase)
            cmd = cal_stance_control_command(desired_k, desired_b, desired_qe)
            pros_state_bf[:] = np.array([
                q_knee_smooth_buffer[-1], q_ankle_smooth_buffer[-1],
                desired_k[idx_knee], desired_b[idx_knee], desired_qe[idx_knee],
                desired_k[idx_ankle], desired_b[idx_ankle], desired_qe[idx_ankle],
                phase
            ])
            debug_ptr += 1
            if debug_ptr == np.shape(data_example)[0]:
                debug_ptr = 0
            time.sleep(1e-2)
            print("\rq_knee ="+format(q_qv[0],'>6.2f')+
                ", q_ankle ="+format(q_qv[1],'>6.2f')+
                ", phase ="+str(phase)+
                ", Kp="+str(np.round(desired_k,2))+
                ", Kb="+str(np.round(desired_b,2))+
                ", qe="+str(np.round(desired_qe,2)),end="")
            
    except KeyboardInterrupt:
        print("Over")