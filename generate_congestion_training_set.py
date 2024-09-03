import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Process

# 获取文件路径列表
def get_sub_path(path):
    sub_path = []
    if isinstance(path,list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p,file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path,file))
    return sub_path

# 将输入缩放到256*256
def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input,(256/dimension[0],256/dimension[1]),order=3)
    return result

# 划分切片
def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

# cv2缩放
def resize_cv2(input):
    output = cv2.resize(input, (256, 256), interpolation = cv2.INTER_AREA)
    return output
# 标准化
def std(input):
    if input.max()==0:
        return input
    else:
        result = (input - input.min())/(input.max()-input.min())
        return result
    
# 保存为npy
def save_npy(out_list,save_path,name):
    output = np.array(out_list)
    output = np.transpose(output,(1,2,0))
    np.save(os.path.join(save_path,name),output)

#
def pack_data(args,name_list,read_features_list,read_label_list,save_path):
    os.system("mkdir -p %s" % (save_path))
    features_save_path = os.path.join(args.save_path,args.task,'feature')
    os.system("mkdir -p %s " % (features_save_path))
    label_save_path = os.path.join(args.save_path,args.task,'lable')
    os.system("mkdir -p %s" % (label_save_path))

    
    for name in name_list:
        out_features_list = []
        for features_name in read_features_list:
            name = os.path.basename(name)
            features = np.load(os.path.join(args.data_path, features_name, name))
            features = std(resize(features))
            out_features_list.append(features)

        save_npy(out_features_list, features_save_path, name)

        out_label_list = []
        congestion_temp = np.zeros((256, 256))
        for label_name in read_label_list:
            name = os.path.basename(name)
            label = np.load(os.path.join(args.data_path, label_name, name))
            congestion_temp += resize(label)

        out_label_list.append(std(congestion_temp))
        save_npy(out_label_list, label_save_path, name)

def parse_args():
    description = "you should add those parameter" 
    parser = argparse.ArgumentParser(description=description)
                                                             
    parser.add_argument("--task", default='congestion', type=str, help='task must be congestion')
    parser.add_argument("--data_path", default='./', type=str, help='path to the decompressed dataset')
    parser.add_argument("--save_path", default='./training_set', type=str, help='path to save training set')

    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args = parse_args()

    features_list = [
        'routability_features/macro_map',
        'routability_features/cell_density',
        'routability_features/net_density/net_density_horizontal',
        'routability_features/net_density/net_density_vertical',
        'routability_features/RUDY/RUDY',
        'routability_features/RUDY/RUDY_pin'
        
    ]

    label_list = [
        'routability_features/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow', 
        'routability_features/congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow',
        'routability_features/congestion/congestion_global_routing/utilization_based/congestion_GR_horizontal_util',
        'routability_features/congestion/congestion_global_routing/utilization_based/congestion_GR_vertical_util'
    ]

    name_list = get_sub_path(os.path.join(args.data_path, features_list[-1]))
    print('processing %s files' % len(name_list))
    save_path = os.path.join(args.save_path, args.task)

    nlist = divide_list(name_list, 1000)
    process = []
    for list in nlist:
        p = Process(target=pack_data, args=(args, list, features_list, label_list, save_path))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
