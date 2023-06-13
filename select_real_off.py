import cv2
import os
import pandas as pd

def choose_data(data_comp):
    if data_comp == False:
        path = r"/home/bestlab/Desktop/shinja_datasets/0809_Sherry_nomask"
        thermalfile = os.listdir(path)
        thermalfile.sort(key=lambda x:int(x.replace("data_","").split('_')[0]))
    else:
        path = r"/media/bestlab/Transcend/Datasets/Dark/0104_shints/Front/Thermal"#0105_yee_Thermal,0104_shints/Front/Thermal
        thermalfile = os.listdir(path)
        thermalfile.sort(key=lambda x:int(x.split('_')[0]))
        # print(thermalfile)
    return path,thermalfile

def real_off(realtime_flag,path,timer,thermal_frame_count,thermalfile,max_thermalframe,data_comp):
    if realtime_flag:
            data = q.get(True, 500)
            timer += 1
    else:
        source = f'{path}/{thermalfile[thermal_frame_count]}'
        if thermal_frame_count!=max_thermalframe:
            df = pd.read_csv(source, header=None)
            if data_comp == False:
                df['clear'] = df.loc[:, 79].str[0:-1]
                df = df.drop(df.columns[[79]], axis=1)
                df_rename = df.rename({'clear': '79'}, axis='columns')
                df_rename = df_rename.to_numpy()
                a = df_rename[59, 79]
                a = a[0:-1]
                df_rename[59, 79] = a
                b = df_rename[0, 0]
                b = b[1:]
                df_rename[0, 0] = b
                df_rename = pd.DataFrame(df_rename)
                df_rename = df_rename.astype(str).astype(int)
                df_rename = df_rename.to_numpy()
                data = df_rename.astype('uint16')#key
                data = cv2.flip(data, 1)
            else:
                df = df.astype(str).astype(int)
                df = df.to_numpy()
                data = df.astype('uint16')
                data = cv2.flip(data, 0)
            thermal_frame_count += 1
    return data,thermal_frame_count