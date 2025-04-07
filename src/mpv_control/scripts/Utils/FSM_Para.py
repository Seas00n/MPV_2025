import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

ser_port_motor = "/dev/ttyUSB1"
ser_port_six = "/dev/ttyUSB0"

# Mode
idx_mode_level_ground = 0
idx_mode_up_slope = 1
idx_mode_down_slope = 2
idx_mode_up_stair = 3
idx_mode_down_stair = 4
# Joint
idx_knee = 0
idx_ankle = 1
# Phase
idx_early_stance = 0
idx_push_off= 1
idx_swing_flexion = 2
idx_swing_extension = 3


pros_log_path = parent_dir+"/pros_log/"
six_force_buffer_path = pros_log_path+"six_force_data.npy"
pros_state_buffer_path = pros_log_path+"pros_state_data.npy" # qk, qa, kpk, kbk, qek, kpa, kba, qea, phase
data_save_path = "/home/yuxuan/Desktop/Huang307/"