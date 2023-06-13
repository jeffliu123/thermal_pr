from cv2 import circle, waitKey
import cv2
import numpy as np
import pandas as pd
import detect_face
import copy
from scipy import signal
import matplotlib.pyplot as plt
import select_real_off

sliding_window = 8
win = [0]*sliding_window

def calculate_snr(max_index,fft_result_real):
    signal_sum = 0
    noise_sum = np.sum(fft_result_real)
    lower_bound = max_index-2#6
    upper_bound = max_index+3#7
    for i in range(lower_bound,upper_bound):
        try:
            if i < 0:
                continue
            else:
                # print(fft_result_real[i])
                signal_sum += fft_result_real[i]
                noise_sum -= fft_result_real[i]
        except IndexError:
            if i > max_index:
                break
    snr = signal_sum/noise_sum
    return snr

def find_kth_largest_index(fft_result_real,n):
    sorted_indices = np.argsort(fft_result_real)
    kth_largest_index = sorted_indices[-n]
    return kth_largest_index


def MOV_AVG_PR(prev_hr_result):
    global sliding_window
    global win
    sum = 0
    count = 0
    win[:-1] = win[1:]
    win[-1] = prev_hr_result
    for i in range(0,len(win)):
        sum += win[i]
    for i in range(0,len(win)):
        if win[i] != 0:
            count += 1
    # print('count: ',count)
    return sum/count

def draw_fft(fft_result_real):
    plt.plot(fft_result_real)
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.show()

def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def main():
    # path = r"/media/bestlab/BEST_1/Sub1/noMask/frames_thermal.csv"
    path_csv = r"/home/bestlab/Desktop/shoot_moon/PR_RR_csv/"
    # df = pd.read_csv(path, header=None)s
    frame_count = 0
    avg_data_list = []
    sampling_rate = 26/3
    FFT_size = 1024
    padded_length = FFT_size
    data_comp = True
    realtime_flag = False
    lock = True
    thermal_frame_count = 0
    timer = 0
    path,thermalfile = select_real_off.choose_data(data_comp)
    max_thermalframe = len(thermalfile)
    global prev_result
    csv_list = []
    # while frame_count < len(df):
    while True:
        data,thermal_frame_count  = select_real_off.real_off(realtime_flag,path,timer,thermal_frame_count,thermalfile,max_thermalframe,data_comp)
        # print(frame_count)
        # data = df.iloc[frame_count][:]
        # data = data.to_numpy()
        # data = data.reshape(120,160)
        data = cv2.resize(data[:,:].astype('uint16'), (640, 480))
        data = cv2.flip(data, 1)
        raw_data = copy.deepcopy(data)
        img = raw_to_8bit(data)
        print(thermal_frame_count)
        # if frame_count == 0:
        if thermal_frame_count == 1:
            rects = detect_face.detect_thermal(img)
            thermal_detect_roi = rects[0][0], rects[0][1], rects[0][2], rects[0][3]
            tracker = cv2.TrackerKCF_create()
            tracker.init(img, thermal_detect_roi)
        # if frame_count >= 1:
        if thermal_frame_count >= 2:
            ok,bbox = tracker.update(img)
            if ok:
                p3 = (int(bbox[0]), int(bbox[1]))
                p4 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])+5)
                s_x_cor = int(bbox[0]-bbox[2]/4)
                s_y_cor = int(bbox[1]+bbox[3]+5)
                s_width = int(bbox[2]*3/2)
                s_height = 100
                neck_x_cor = int(bbox[0]+1/3*bbox[2])
                neck_y_cor = int(bbox[1]+bbox[3]+1/8*bbox[3])
                neck_width = 60
                neck_height = 30
            avg_data = np.mean(raw_data[neck_y_cor:neck_y_cor+neck_height,neck_x_cor:neck_x_cor+neck_width])
            # avg_data = np.mean(img[neck_y_cor:neck_y_cor+neck_height,neck_x_cor:neck_x_cor+neck_width])
            avg_data = round(avg_data,2)
            # print(avg_data)
            avg_data_list.append(avg_data)
            if len(avg_data_list) > 128:
                avg_data_list.pop(0)
            if len(avg_data_list) == 128:
                # if (frame_count-128)%9 == 0:
                # if (thermal_frame_count-128)%9 == 0:
                avg_data_array = np.array(avg_data_list)
                b, a = signal.butter(4, [0.8, 1.8], 'bandpass', fs=sampling_rate)#1.33,1.5
                filtedData = signal.filtfilt(b, a, avg_data_array)#15
                filtedData = filtedData*75
                b, a = signal.butter(4, [1, 1.67], 'bandpass', fs=sampling_rate)
                filtedData = signal.filtfilt(b, a, filtedData)
                filtedData = filtedData*120
                padding = padded_length - len(filtedData)
                padded_signal = np.pad(filtedData,(0,padding),'constant')
                fft_result = np.fft.fft(padded_signal)
                fft_result_real = np.real(fft_result)
                fft_result_real = fft_result_real[0:FFT_size//2]
                fft_result_real = np.abs(fft_result_real)
                # fft_result_real = fft_result_real[119:200]#60,110
                max_index = np.argmax(fft_result_real)
                snr = calculate_snr(max_index,fft_result_real)
                second_largest_index = find_kth_largest_index(fft_result_real,2)
                snr1 = calculate_snr(second_largest_index,fft_result_real)
                third_largest_index = find_kth_largest_index(fft_result_real,3)
                snr2 = calculate_snr(third_largest_index,fft_result_real)
                print('snr: ',snr,'snr1: ',snr1,'snr2: ',snr2)
                print('=======')
                print('max_index: ',max_index,'second_largest_index: ',second_largest_index,'third_largest_index: ',third_largest_index)
                print('=======')
                PR_result = sampling_rate/FFT_size*(max_index+100)*60
                PR_result = round(PR_result,0)
                PR_result1 = sampling_rate/FFT_size*(second_largest_index+100)*60
                PR_result1 = round(PR_result1,0)
                PR_result2 = sampling_rate/FFT_size*(third_largest_index+100)*60
                PR_result2 = round(PR_result2,0)
                print('---->>',PR_result)
                print('---->>',PR_result1)
                print('---->>',PR_result2)
            # if (frame_count-128)%9 == 0:
            # if (thermal_frame_count-128)%9 == 0:
                if lock == True:
                    prev_result = PR_result
                    lock = False
                else:
                    if abs(PR_result-prev_result)>15:
                        PR_result = prev_result
                    else:
                        prev_result = PR_result
                final_result = MOV_AVG_PR(PR_result)
                final_result = round(final_result,0)
                print('--------->>',final_result)
                csv_list.append(final_result)
                # print('--------->>',PR_result)
                # print(max_index)
                plt.ion()
                draw_fft(fft_result_real)
            # if frame_count == 1578:
            if thermal_frame_count == 744:
                data_PR = pd.DataFrame(csv_list)
                data_PR.to_csv(path_csv+'PR_shints_paper.csv', index=False, sep= ",", header=None )
                print('success!!!!')

            cv2.rectangle(img, p3, p4, (0, 0, 255), 2, 1)
            cv2.rectangle(img, (s_x_cor,s_y_cor), (s_x_cor+s_width,s_y_cor+s_height), (0,0,255), 2, 1)
            cv2.rectangle(img, (neck_x_cor,neck_y_cor), (neck_x_cor+neck_width,neck_y_cor+neck_height), (0,0,255), 2, 1)
        cv2.imshow('thermal',img)
        cv2.waitKey(1)
        frame_count +=1
        plt.cla()
        plt.ioff()

main()    