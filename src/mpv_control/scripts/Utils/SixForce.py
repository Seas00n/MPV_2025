import numpy as np
import serial
import time
import struct
from FSM_Para import six_force_buffer_path, ser_port_six

try:
    ser = serial.Serial(
            port=ser_port_six,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS, timeout=1e-2)
    if ser.is_open:
        print("\033[32m [LOG]Serial {} is open\033[0m".format(ser))
        # set_update_rate = "AT+GSD=STOP\r\n".encode('utf-8')
        # print(set_update_rate)
        # ser.write(bytearray(set_update_rate))
        # d_ = ser.read_all()
        # print(d_.decode('utf-8'))
    else:
        print("\033[31m [ERROR]Serial not Open \033[0m")
except KeyboardInterrupt as e:
    print("\033[31m [ERROR]Serial not Open | {}\033[0m".format(e))
time.sleep(1)  # sleep 100 ms




def init_force():
    ##################################################
    set_update_rate = "AT+SMPF=300\r\n".encode('utf-8')
    print(set_update_rate)
    ser.write(bytearray(set_update_rate))
    time.sleep(0.1)
    d_ = ser.read_all()
    print(d_.decode('utf-8'))
    ##################################################
    set_recieve_format = "AT+SGDM=(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)\r\n".encode('utf-8')
    print(set_recieve_format)
    ser.write(bytearray(set_recieve_format))
    time.sleep(0.1)
    d_ = ser.read_all()
    print(d_.decode('utf-8'))
    ##################################################
    get_data_once = "AT+GSD\r\n".encode('utf-8')
    ser.write(bytearray(get_data_once))
    init_f=[]
    fx_init = 0
    fy_init = 0
    fz_init = 0
    mx_init = 0
    my_init = 0
    mz_init = 0
    j = 0
    catch_pkg = False
    CmdPacket_Begin_Buffer = np.zeros((31,)).tolist()
    init_num = 0
    while init_num < 50:
        while not catch_pkg:
            data = ser.read()
            if len(data)>0:
                data = np.frombuffer(data, dtype=np.uint8)[0]
                CmdPacket_Begin_Buffer[0:-1] = CmdPacket_Begin_Buffer[1:]
                CmdPacket_Begin_Buffer[-1] = data
                print(CmdPacket_Begin_Buffer[0], CmdPacket_Begin_Buffer[1])
            if CmdPacket_Begin_Buffer[0] == 0xAA and CmdPacket_Begin_Buffer[1]==0x55:
                catch_pkg = True
                print("Catch Pkg")
                continue
            
        data = ser.read(31)
        if data[0] == 0xAA and data[1] == 0x55 and len(data)==31:
             f_data = np.frombuffer(data[6:30], dtype=np.dtype('<f4'))
             fx_init += f_data[0]
             fy_init += f_data[1]
             fz_init += f_data[2]
             mx_init += f_data[3]
             my_init += f_data[4]
             mz_init += f_data[5]
             init_num += 1
             print("Init Num", init_num)
        time.sleep(1e-2)


    # for i in range(50):
    #     init_data = ser.readall()
    #     if len(init_data) > 30:
    #         if 0xAA == init_data[0] and 0x55 == init_data[1]:
    #             fx_init = struct.unpack('f', init_data[6:10])[0] + fx_init
    #             fy_init = struct.unpack('f', init_data[10:14])[0] + fy_init
    #             fz_init = struct.unpack('f', init_data[14:18])[0] + fz_init
    #             mx_init = struct.unpack('f', init_data[18:22])[0] + mx_init
    #             my_init = struct.unpack('f', init_data[22:26])[0] + my_init
    #             mz_init = struct.unpack('f', init_data[26:30])[0] + mz_init
    #             j = j + 1
    #     else:
    #         print("\rWait for data.", end="")
    #     time.sleep(5e-2)
    j = init_num
    fx_init = fx_init / j
    fy_init = fy_init / j
    fz_init = fz_init / j
    mx_init = mx_init / j
    my_init = my_init / j
    mz_init = mz_init / j
    init_f[0:6] = fx_init, fy_init, fz_init, mx_init, my_init, mz_init
    print("Init Over")
    return init_f

if __name__ == "__main__":
    init_f = init_force()
    F_memmap = np.memmap(
        six_force_buffer_path,
        dtype='float32',
        mode='w+',
        shape=(6,)
    )
    fff = np.zeros((6,))
    np.set_printoptions(precision=3, suppress=True)
    try:
        catch_pkg = False
        CmdPacket_Begin_Buffer = np.zeros((31,)).tolist()
        while True:
            while not catch_pkg:
                data = ser.read()
                if len(data)>0:
                    data = np.frombuffer(data, dtype=np.uint8)[0]
                    CmdPacket_Begin_Buffer[0:-1] = CmdPacket_Begin_Buffer[1:]
                    CmdPacket_Begin_Buffer[-1] = data
                if CmdPacket_Begin_Buffer[0] == 0xAA and CmdPacket_Begin_Buffer[1]==0x55:
                    catch_pkg = True
                    print("Catch Pkg")
                    continue
            # data = ser.read(31*5)
            # 如果用自带的线就readall
            data = ser.readall()
            if len(data) > 30 and 0xAA == data[0] and 0x55 == data[1]:
                print(data[0], data[1])
                fx = struct.unpack('f', data[6:10])[0] - init_f[0]
                fy = struct.unpack('f', data[10:14])[0] - init_f[1]
                fz = struct.unpack('f', data[14:18])[0] - init_f[2]
                mx = struct.unpack('f', data[18:22])[0] - init_f[3]
                my = struct.unpack('f', data[22:26])[0] - init_f[4]
                mz = struct.unpack('f', data[26:30])[0] - init_f[5]
                fff[0:6] = -fx, fy, -fz, mx, -my, mz
                print(np.round(fff, 2))
                F_memmap[:] = fff
                F_memmap.flush()
                time.sleep(1e-2)
    except KeyboardInterrupt:
        print("Over")
        set_update_rate = "AT+GSD=STOP\r\n".encode('utf-8')
        print(set_update_rate)
        ser.write(bytearray(set_update_rate))
        d_ = ser.read_all()
        print(d_.decode('utf-8'))
        ser.close()
    e