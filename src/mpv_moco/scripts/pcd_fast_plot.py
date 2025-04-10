import matplotlib.pyplot as plt
import matplotlib
from matplotlib import *
import numpy as np
import time

class FastPlotCanvas(object):
    ax: matplotlib.axes._axes.Axes
    fig: matplotlib.figure.Figure
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        self.line_pcd, = self.ax.plot(np.zeros(1, ))
        self.pcd = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='b', alpha=0.1)
        self.fea_A = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='m', linewidths=1)
        self.fea_B = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='g', linewidths=1)
        self.fea_C = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='y', linewidths=1)
        self.fea_D = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='tab:purple', linewidths=1)
        self.fea_E = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='tab:green', linewidths=1)
        self.fea_F = self.ax.scatter(np.zeros(1, ), np.zeros(1, ), color='tab:orange', linewidths=1)
        self.text_info = self.ax.text(0,0,"")
        self.env_para, = self.ax.plot(np.zeros(1, ), color='r', linewidth=1)
        plt.show(block=False)
    
    def update_canvas(self):
        if self.ax.get_xlim() != (-0.2, 1):
            self.ax.set_xlim(-0.2, 1)
        if self.ax.get_ylim() != (-0.6, 0.6):
            self.ax.set_ylim(-0.6, 0.6)
        # self.fig.canvas.update()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def set_pcd(self, pcd_data):
        self.pcd.set_offsets(pcd_data)
        self.ax.draw_artist(self.pcd)
    
    def set_fea_A(self, fea_data):
        self.fea_A.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_A)

    def set_fea_B(self, fea_data):
        self.fea_B.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_B)

    def set_fea_C(self, fea_data):
        self.fea_C.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_C)

    def set_fea_D(self, fea_data):
        self.fea_D.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_D)

    def set_fea_E(self, fea_data):
        self.fea_E.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_E)

    def set_fea_F(self, fea_data):
        self.fea_F.set_offsets(fea_data)
        # self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.fea_F)
    
    def set_info(self, px, py, id, corner_situation, env_rotate):
        self.text_info.set_text("id:{},corner_situation:{},env_rotate:{}".format(id,
                            round(corner_situation, 2), round(env_rotate, 2)))
        self.text_info.set_x(px)
        self.text_info.set_y(py)
        self.ax.draw_artist(self.text_info)
    
    def set_env_paras(self, xc, yc, w, h, p=None):
        if p is None:
            p = [0, 0]
        x = np.array([xc - w + p[0], xc + p[0], xc + p[0], xc + w + p[0]])
        y = np.array([yc - h + p[1], yc - h + p[1], yc + p[1], yc + p[1]])
        self.env_para.set_xdata(x)
        self.env_para.set_ydata(y)
        self.ax.draw_artist(self.env_para)

    def close(self):
        plt.close(self.fig)
    
if __name__ == "__main__":
    fpc = FastPlotCanvas()
    tstart = time.time()
    num_frames = 0

    while time.time()-tstart < 5:
        pcd_data = np.random.random((100,2))
        pcd_data[:,0] = np.linspace(-1,1,np.shape(pcd_data)[0])
        fpc.set_pcd(pcd_data)
        fpc.set_fea_A(np.random.random((10,2)))
        fpc.set_fea_B(np.random.random((10,2)))
        fpc.set_info(0,-0.2,1,1,1)
        fpc.update_canvas()
        num_frames += 1
    fpc.close()
    print(num_frames/5)