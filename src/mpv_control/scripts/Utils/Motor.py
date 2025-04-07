import numpy as np
import serial

send_save_list = []

def init_motor(port):
    ser = serial.Serial(
        port=port,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS, timeout=1e-3)
    return ser

def send_signals_to_motor(ser, q_qv_d):
    byte_vec = bytearray(q_qv_to_motor_signal(q_qv_d))
    # print("send:"+byte_vec.hex())
    ser.write(byte_vec)
    # try:
    #     ser.write(byte_vec)
    # except Exception as e:
    #     print(byte_vec)
    #     print(e)
    #     return

def read_signals_from_motor(ser:serial.Serial, data_byte_size=16):
    # read_byte_vec = ser.read(data_byte_size)
    # print("read"+read_byte_vec.hex())
    # q_qv_motor = np.frombuffer(read_byte_vec, dtype=np.uint16)
    # if len(q_qv_motor) == 8:
    #     q_qv = motor_signal_to_q_qv(q_qv_motor[0:4])
    #     i_t = motor_signal_to_q_qv(q_qv_motor[4:8])
    # else:
    #     print('The data length {} is incorrect!'.format(len(read_byte_vec)))
    #     q_qv = np.zeros(4)
    #     i_t = np.zeros(4)
    # return q_qv,i_t
    try:
        read_byte_vec = ser.read(data_byte_size)
        # print("read:"+read_byte_vec.hex())
        q_qv_motor = np.frombuffer(read_byte_vec, dtype=np.uint16)
        if len(q_qv_motor) == 8:
            q_qv = motor_signal_to_q_qv(q_qv_motor[0:4])
            i_t = motor_signal_to_q_qv(q_qv_motor[4:8])
        else:
            print('The data length {} is incorrect!'.format(len(read_byte_vec)))
            q_qv = np.zeros(4)
            i_t = np.zeros(4)
        return q_qv,i_t
    except Exception as e:
        print(e)
        q_qv = np.zeros(4)
        i_t = np.zeros(4)
        return q_qv,i_t
    

def q_qv_to_motor_signal(q_qv_d, k_float_2_int=100, b_float_2_int=30000):
    return (k_float_2_int * q_qv_d + b_float_2_int).astype(np.uint16)  # 100 * (q_k_d, qv_k_d, q_a_d,qv_a_d)


def motor_signal_to_q_qv(q_qv, k_float_2_int=100, b_float_2_int=30000):
    return (q_qv.astype(float) - b_float_2_int) / k_float_2_int  # 100 * (q_k_d, qv_k_d, q_a_d,qv_a_d)
