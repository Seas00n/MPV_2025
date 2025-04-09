import sys
import os
from PyQt5 import QtCore
import pyqtgraph as pg
import numpy as np
import lcm
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, QVBoxLayout, QWidget
import pyqtgraph.parametertree as ptree
from pyqtgraph.parametertree import interact, ParameterTree, Parameter
current_dir = os.path.dirname(os.path.realpath(__file__))
mpv_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.append(mpv_dir+"/mpv_moco/scripts/")
from core.impedance_lcm.exlcm import impedance_info
from Utils.FSM_Para import data_save_path


k0 = [[100, 100, 20, 15], [75, 90, 30, 30]] # [[70, 70, 20, 15], [75, 90, 25, 30]]
b0 = [[5, 10, 1.05, 2.05], [5, 5, 1, 1]]#[[5, 5, 2, 2], [5, 5, 1, 1]]
q_e0 = [[7, 12, 75, 2], [5, 12, -2, 1]]



lim_kp = [
    [[70.,140.],[70.,140.],[15.,30.],[15, 25.]], #kp lim for knee
    [[70.,140.],[70.,140.],[10.,40.],[10.,30.]] #kp lim for ankle
]


lim_kb = [
    [[8,12], [8,12], [1,2], [1.2,2.5]], #kb lim for knee
    [[1,10], [1,10], [1,2], [1,3]] #kp lim for ankle
]


lim_qe = [
    [[0., 25.], [2., 25.], [40., 80.], [0., 35.]], #qe lim for knee
    [[-5., 10.], [5., 14.], [-9., -2], [-5., 5.]] #qe lim for ankle
]

env_mode = "slope"

lc = lcm.LCM()

def normalize_imp(imp,lim):
    return int((imp-lim[0])/(lim[1]-lim[0])*100)



def imp_interpolate(task1, task2):
    if env_mode == "slope":
        data = {
            (0.6, 0): [86 ,89 ,18 ,15,89.6,89.6,25 ,25 ,11.5,8.5,1.39,1.99 ,5.3,8.48,2.6,2 ,8 ,16.72,59,1.95,3.15,5,-2.07,3.15],
            (0.6, 2): [95.9,93.1,17.55,15.0,89.6,74.9,22.3,22.4,8.5,8.5,1.29,1.993,8.5,8.5,1.0,1.0,13.5,11.2,55.0,5.4,5.54,7.43,-2.07,5.54],
            (0.6, 4): [96.6,97.3,19.95,15.0,89.6,74.9,22.3,22.4,9.35,8.505,1.51,1.993,8.5,8.7,1.0,1.0,7.5,14.88,55.0,0.0,5.925,7.61,-2.07,5.84],
            (0.8, 0): [89.6,89.6,17.25,15.0,89.6,89.6,25.0,25.0,10.0,8.5,1.64,1.993,6.9,8.49,1.8,1.5,9.5,15.8,55,1.95,2.78,6.89,-5.01,2.78],
            (0.8, 2): [100.0,100.0,21.59,15.89,87.5,87.6,24.7,25.594,8.266,8.32,1.26,1.577,9.469,9.273,1.0,1.0,13.0,16.11,64.0,5.9,1.05,6.891,-2.07,1.714],
            (0.8, 4): [100.1,101.5,19.95,15.4,89.6,89.6,25.0,25.0,9.0,8.5,1.3,1.993,8.5,8.5,1.0,1.0,12.0,14.88,55.25,1.95,2.86,8.51,-2.07,2.86],
            (1.0, 0): [92.4,91.0,16.5,16.4,89.6,89.6,25.0,25.0,8.5,8.5,1.12,1.993,8.5,8.5,1.0,1.0,11.0,15.11,55.0,1.95,2.41,9.14,-2.07,2.41],
            (1.0, 2): [106.4,105.0,17.85,16.9,89.6,105.0,27.4,27.4,8.5,8.5,1.1,2.227,8.5,8.5,1.0,1.0,16.0,14.19,55.0,8.4,-0.72,10.04,-2.07,-0.72],
        }
    elif env_mode == "stair":
        # TODO:
        data = {
            (0.6, 0): [86 ,89 ,18 ,15,89.6,89.6,25 ,25 ,11.5,8.5,1.39,1.99 ,5.3,8.48,2.6,2 ,8 ,16.72,59,1.95,3.15,5,-2.07,3.15],
            (0.6, 2): [95.9,93.1,17.55,15.0,89.6,74.9,22.3,22.4,8.5,8.5,1.29,1.993,8.5,8.5,1.0,1.0,13.5,11.2,55.0,5.4,5.54,7.43,-2.07,5.54],
            (0.6, 4): [96.6,97.3,19.95,15.0,89.6,74.9,22.3,22.4,9.35,8.505,1.51,1.993,8.5,8.7,1.0,1.0,7.5,14.88,55.0,0.0,5.925,7.61,-2.07,5.84],
            (0.8, 0): [89.6,89.6,17.25,15.0,89.6,89.6,25.0,25.0,10.0,8.5,1.64,1.993,6.9,8.49,1.8,1.5,9.5,15.8,55,1.95,2.78,6.89,-5.01,2.78],
            (0.8, 2): [100.0,100.0,21.59,15.89,87.5,87.6,24.7,25.594,8.266,8.32,1.26,1.577,9.469,9.273,1.0,1.0,13.0,16.11,64.0,5.9,1.05,6.891,-2.07,1.714],
            (0.8, 4): [100.1,101.5,19.95,15.4,89.6,89.6,25.0,25.0,9.0,8.5,1.3,1.993,8.5,8.5,1.0,1.0,12.0,14.88,55.25,1.95,2.86,8.51,-2.07,2.86],
            (1.0, 0): [92.4,91.0,16.5,16.4,89.6,89.6,25.0,25.0,8.5,8.5,1.12,1.993,8.5,8.5,1.0,1.0,11.0,15.11,55.0,1.95,2.41,9.14,-2.07,2.41],
            (1.0, 2): [106.4,105.0,17.85,16.9,89.6,105.0,27.4,27.4,8.5,8.5,1.1,2.227,8.5,8.5,1.0,1.0,16.0,14.19,55.0,8.4,-0.72,10.04,-2.07,-0.72],
        }
    for k in data: data[k] = np.array(data[k])

    if (task1, task2) in data:
        return data[(task1, task2)]

    t1_vals = sorted(set(k[0] for k in data))
    t2_vals = sorted(set(k[1] for k in data))
    t1a, t1b = max(x for x in t1_vals if x <= task1), min(x for x in t1_vals if x >= task1)
    t2a, t2b = max(x for x in t2_vals if x <= task2), min(x for x in t2_vals if x >= task2)

    try:
        Q11, Q12 = data[(t1a, t2a)], data[(t1a, t2b)]
        Q21, Q22 = data[(t1b, t2a)], data[(t1b, t2b)]
    except KeyError:
        raise ValueError("Interpolation failed due to missing corner points.")

    t = (task1 - t1a) / (t1b - t1a)
    u = (task2 - t2a) / (t2b - t2a)

    return (1 - t) * (1 - u) * Q11 + t * (1 - u) * Q21 + (1 - t) * u * Q12 + t * u * Q22



class MySlider(QWidget):
    def __init__(self, name, v0, min, max, parent=None):
        super(MySlider, self).__init__(parent=parent)
        self.label = QLabel(self)
        self.val = v0
        self.name = name
        self.label.setText(name+":{:2g}".format(self.val))
        self.hLayout = QHBoxLayout(self)
        self.hLayout.addWidget(self.label)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.hLayout.addWidget(self.slider)
        # spaceItem = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        # self.hLayout.addItem(spaceItem)
        self.resize(self.sizeHint())
        self.min = min
        self.max = max
        self.slider.valueChanged.connect(self.set_Label)
        self.slider.setValue(normalize_imp(self.val, [self.min, self.max]))
    
    def set_Label(self, value):
        self.val = value*0.01*(self.max-self.min)+self.min
        self.label.setText(self.name+":{:2g}".format(self.val))



class TaskTuner(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800,600)
        self.layout = pg.LayoutWidget()
        self.setCentralWidget(self.layout)
        
        self.k_mat = np.array(k0)
        self.b_mat = np.array(b0)                           
        self.q_e_mat = np.array(q_e0)

        self.learning_evaluation = 0
        
        tree = ParameterTree()
        tree.setMinimumWidth(200)
        if env_mode == "slope":
            self.v = 0.6
            self.s = 0.0
            v_key = ["0.6", "0.8", "1.0"]
            s_key = ["0", "2", "4"]
            para1_type = dict(type='list', values=v_key)
            para2_type = dict(type='list', values=s_key)
            contextParam = interact(self.next_data, task1 = para1_type, task2 = para2_type)
            contextParam.setName("1v, 2s")
        elif env_mode == "stair":
            self.w = 0.2
            self.h = 0.2
            w_key = ["0.6", "0.8", "1.0"]
            h_key = ["0", "2", "4"]
            para1_type = dict(type='list', values=w_key)
            para2_type = dict(type='list', values=h_key)
            contextParam = interact(self.next_data, task1 = para1_type, task2 = para2_type)
            contextParam.setName("1w, 2h")
        tree.addParameters(contextParam, showTop=True)
        self.params = ptree.Parameter.create(name='Parameters', type='group', children=[
            dict(name='idx_imp',type='int', value=0, limits=[0,30])
        ])
        self.params.children()[0].sigValueChanged.connect(self.change_tp)
        tree.addParameters(self.params)
        sendParam = interact(self.send_imp)
        tree.addParameters(sendParam, showTop=True)
        saveParam = interact(self.save_imp)
        tree.addParameters(saveParam, showTop=True)
        self.layout.addWidget(tree, row=0, col=0)
        self.init_slider()
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.init_win()
        self.layout.addWidget(self.win, row=1, col=0, rowspan=1, colspan=3)

    def next_data(self, task1='0.6', task2='0'):
        if env_mode == "slope":
            self.v = float(task1)
            self.s = float(task2)
            imp_task = imp_interpolate(self.v, self.s)
        elif env_mode == "stair":
            self.w = float(task1)
            self.h = float(task2)
            imp_task = imp_interpolate(self.w, self.h)
        
        self.k_mat = imp_task[0:8].reshape((2,4))
        self.b_mat = imp_task[8:16].reshape((2,4))
        self.q_e_mat = imp_task[16:].reshape((2,4))
        self.set_slider_pos()
        self.reward_k[:] = 0
        self.reward_t[:] = 0
        self.ax_rk.setData(y=self.reward_k)
        self.ax_rt.setData(y=self.reward_t*0.5+self.reward_t*0.5)
    
    def send_imp(self):
        self.set_imp_from_slider()
        msg = impedance_info()
        imp_send = np.hstack([self.k_mat.flatten(),self.b_mat.flatten(),self.q_e_mat.flatten()]).reshape((1,-1))
        imp_send = imp_send.reshape((6,4)).T[:,[0,2,4,1,3,5]]
        msg.para_phase1 = imp_send[0,:]
        msg.para_phase2 = imp_send[1,:]
        msg.para_phase3 = imp_send[2,:]
        msg.para_phase4 = imp_send[3,:]
        lc.publish("Impedance_Info", msg.encode())

    def save_imp(self):
        print("Imp has been saved")
        imp_send = np.hstack([self.k_mat.flatten(),self.b_mat.flatten(),self.q_e_mat.flatten()]).reshape((1,-1))
        if env_mode == "slope":
            save_path = data_save_path+"/v_{}_s_{}.npy".format(self.v, self.s)
            # np.save(save_path, imp_send)
        elif env_mode == "stair":
            save_path = data_save_path+"/w_{}_h_{}.npy".format(self.w, self.h)
            # np.save(save_path, imp_send)

    def set_imp_from_slider(self):
        self.k_mat[0][0] = self.skp_k1.val
        self.k_mat[0][1] = self.skp_k2.val
        self.k_mat[0][2] = self.skp_k3.val
        self.k_mat[0][3] = self.skp_k4.val
        self.b_mat[0][2] = self.skb_k3.val
        self.b_mat[0][3] = self.skb_k4.val
        self.q_e_mat[0][0] = self.sqe_k1.val
        self.q_e_mat[0][1] = self.sqe_k2.val
        self.q_e_mat[0][2] = self.sqe_k3.val
        self.q_e_mat[0][3] = self.sqe_k4.val
        self.k_mat[1][0] = self.skp_a1.val
        self.k_mat[1][1] = self.skp_a2.val
        self.k_mat[1][2] = self.skp_a3.val
        self.k_mat[1][3] = self.skp_a4.val
        self.q_e_mat[1][1] = self.sqe_a2.val
        self.q_e_mat[1][2] = self.sqe_a3.val

    def init_slider(self):
        w = QWidget()
        vlayout = QVBoxLayout(w)
        #########################
        w1 = QWidget()
        hlayout1 = QHBoxLayout(w1)
        self.skp_k1 = MySlider("膝Kp1",v0=99, min=lim_kp[0][0][0], max=lim_kp[0][0][1])
        self.skp_k2 = MySlider("膝Kp2",v0=99, min=lim_kp[0][1][0], max=lim_kp[0][1][1])
        self.skp_k3 = MySlider("膝Kp3",v0=99, min=lim_kp[0][2][0], max=lim_kp[0][2][1])
        self.skp_k4 = MySlider("膝Kp4",v0=99, min=lim_kp[0][3][0], max=lim_kp[0][3][1])
        hlayout1.addWidget(self.skp_k1)
        hlayout1.addWidget(self.skp_k2)
        hlayout1.addWidget(self.skp_k3)
        hlayout1.addWidget(self.skp_k4)
        vlayout.addWidget(w1)
        #########################
        w2 = QWidget()
        hlayout2 = QHBoxLayout(w2)
        self.skb_k3 = MySlider("膝Kb3",v0=99, min=lim_kb[0][2][0], max=lim_kb[0][2][1])
        self.skb_k4 = MySlider("膝Kb4",v0=99, min=lim_kb[0][3][0], max=lim_kb[0][3][1])
        hlayout2.addWidget(self.skb_k3)
        hlayout2.addWidget(self.skb_k4)
        vlayout.addWidget(w2)
        #######################
        w3 = QWidget()
        hlayout3 = QHBoxLayout(w3)
        self.sqe_k1 = MySlider("膝qe1",v0=99, min=lim_qe[0][0][0], max=lim_qe[0][0][1])
        self.sqe_k2 = MySlider("膝qe2",v0=99, min=lim_qe[0][1][0], max=lim_qe[0][1][1])
        self.sqe_k3 = MySlider("膝qe3",v0=99, min=lim_qe[0][2][0], max=lim_qe[0][2][1])
        self.sqe_k4 = MySlider("膝qe4",v0=99, min=lim_qe[0][3][0], max=lim_qe[0][3][1])
        hlayout3.addWidget(self.sqe_k1)
        hlayout3.addWidget(self.sqe_k2)
        hlayout3.addWidget(self.sqe_k3)
        hlayout3.addWidget(self.sqe_k4)
        vlayout.addWidget(w3)
        #######################
        w4 = QWidget()
        hlayout4 = QHBoxLayout(w4)
        self.skp_a1 = MySlider("踝Kp1",v0=99, min=lim_kp[1][0][0], max=lim_kp[1][0][1])
        self.skp_a2 = MySlider("踝Kp2",v0=99, min=lim_kp[1][1][0], max=lim_kp[1][1][1])
        self.skp_a3 = MySlider("踝Kp3",v0=99, min=lim_kp[1][2][0], max=lim_kp[1][2][1])
        self.skp_a4 = MySlider("踝Kp4",v0=99, min=lim_kp[1][3][0], max=lim_kp[1][3][1])
        hlayout4.addWidget(self.skp_a1)
        hlayout4.addWidget(self.skp_a2)
        hlayout4.addWidget(self.skp_a3)
        hlayout4.addWidget(self.skp_a4)
        vlayout.addWidget(w4)
        #######################
        w5 = QWidget()
        hlayout5 = QHBoxLayout(w5)
        # self.sqe_a1 = MySlider("踝qe1",v0=99, min=lim_qe[1][0][0], max=lim_qe[1][0][1])
        self.sqe_a2 = MySlider("踝qe2",v0=99, min=lim_qe[1][1][0], max=lim_qe[1][1][1])
        self.sqe_a3 = MySlider("踝qe3",v0=99, min=lim_qe[1][2][0], max=lim_qe[1][2][1])
        # self.sqe_a4 = MySlider("踝qe4",v0=99, min=lim_qe[1][3][0], max=lim_qe[1][3][1])
        # hlayout5.addWidget(self.sqe_a1)
        hlayout5.addWidget(self.sqe_a2)
        hlayout5.addWidget(self.sqe_a3)
        # hlayout5.addWidget(self.sqe_a4)
        vlayout.addWidget(w5)
        #######################
        self.layout.addWidget(w, row=0, col=1, rowspan=1, colspan=2)
        self.set_slider_pos()
    
    def init_win(self):
        p0 = self.win.addPlot()
        p0.setFixedWidth(600)
        p0.setYRange(0,1)
        self.t = np.arange(0,6)
        self.reward_k = np.zeros_like(self.t).astype(np.float32)
        self.reward_t = np.zeros_like(self.t).astype(np.float32)
        self.ax_rk = p0.plot(self.t, self.reward_k,  pen=pg.mkPen(color=(255,0,0,150), width=3), symbol='o')
        self.ax_rt = p0.plot(self.t, self.reward_t,  pen=pg.mkPen(color=(0,255,0,150), width=3), symbol='o')

    def set_slider_pos(self):
        ##############################################################################
        self.skp_k1.slider.setValue(normalize_imp(self.k_mat[0][0], lim=lim_kp[0][0]))
        self.skp_k2.slider.setValue(normalize_imp(self.k_mat[0][1], lim=lim_kp[0][1]))
        self.skp_k3.slider.setValue(normalize_imp(self.k_mat[0][2], lim=lim_kp[0][2]))
        self.skp_k4.slider.setValue(normalize_imp(self.k_mat[0][3], lim=lim_kp[0][3]))
        ##############################################################################
        self.skb_k3.slider.setValue(normalize_imp(self.b_mat[0][2], lim=lim_kb[0][2]))
        self.skb_k4.slider.setValue(normalize_imp(self.b_mat[0][3], lim=lim_kb[0][3]))
        ##############################################################################
        self.sqe_k1.slider.setValue(normalize_imp(self.q_e_mat[0][0], lim=lim_qe[0][0]))
        self.sqe_k2.slider.setValue(normalize_imp(self.q_e_mat[0][1], lim=lim_qe[0][1]))
        self.sqe_k3.slider.setValue(normalize_imp(self.q_e_mat[0][2], lim=lim_qe[0][2]))
        self.sqe_k4.slider.setValue(normalize_imp(self.q_e_mat[0][3], lim=lim_qe[0][3]))
        ##############################################################################
        self.skp_a1.slider.setValue(normalize_imp(self.k_mat[1][0], lim=lim_kp[1][0]))
        self.skp_a2.slider.setValue(normalize_imp(self.k_mat[1][1], lim=lim_kp[1][1]))
        self.skp_a3.slider.setValue(normalize_imp(self.k_mat[1][2], lim=lim_kp[1][2]))
        self.skp_a4.slider.setValue(normalize_imp(self.k_mat[1][3], lim=lim_kp[1][3]))
        ##############################################################################
        # self.sqe_a1.slider.setValue(normalize_imp(self.q_e_mat[1][0], lim=lim_qe[1][0]))
        self.sqe_a2.slider.setValue(normalize_imp(self.q_e_mat[1][1], lim=lim_qe[1][1]))
        self.sqe_a3.slider.setValue(normalize_imp(self.q_e_mat[1][2], lim=lim_qe[1][2]))
        # self.sqe_a4.slider.setValue(normalize_imp(self.q_e_mat[1][3], lim=lim_qe[1][3]))

    def change_tp(self):
        self.learning_evaluation = self.params.child('idx_imp').value()
        self.reward_k[:] = 0
        self.reward_t[:] = 0
        self.ax_rk.setData(y=self.reward_k)
        self.ax_rt.setData(y=self.reward_t*0.5+self.reward_k*0.5)

if __name__ == '__main__':
    pg.mkQApp()
    task_tuner = TaskTuner()
    task_tuner.show()
    pg.exec()