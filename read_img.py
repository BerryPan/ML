# 读取图片信息
import cv2
import math
import numpy as np
anno_path = "FDDB-folds/"
ori_path = "originalPics/"
file_name = "FDDB-fold-01-ellipseList.txt"


def get_img_info():
    f = open(anno_path + file_name)
    data = f.read()
    data = data.split('\n')
    f.close()
    i = 0
    img_dict = {} # img_name:[[r1,r2,angle,x,y],...]
    while i < len(data)-2:
        img_name = data[i]
        img_info = []
        face_num = int(data[i+1])
        for j in range(face_num):
            img_info.append(data[i+2+j].split(' ')[:5])
        img_dict[img_name] = img_info
        i = i+face_num+2
    return img_dict


def gene_pos_img(dic):
    # 生成positive img（拓展1/3框住原椭圆）
    img_lst = list(dic.keys())
    num = 0
    for img_name in img_lst:
        img = cv2.imread(ori_path+img_name+'.jpg')
        for face in dic[img_name]:
            x1, y1, x2, y2 = cal_rac(float(face[0])*2, float(face[1])*2, float(face[2])*180/math.pi,
                                     float(face[3]), float(face[4]))
            # print(x1, x2, y1, y2)
            a = cv2.copyMakeBorder(img, abs(y2-y1), abs(y2-y1), abs(x2-x1), abs(x2-x1), cv2.BORDER_REPLICATE)
            # cv2.imshow('image', a)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # a = a[y2-y1+y1:y2-y1+y2, x2-x1+x1:x2-x1+x2]
            a = a[abs(y2-y1)+min(y1, y2):abs(y2-y1)+max(y1, y2), abs(x2-x1)+min(x1, x2):abs(x2-x1)+max(x1, x2)]
            # a = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            cv2.imwrite("pos/"+str(num)+".jpg", a)
            num += 1
            print(num)


def cal_rac(major, minor, angle_deg, x, y):
    # 计算长方形边界框的两个顶点
    t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
    [max_x, min_x] = [x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) -
                      minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t, t + np.pi)]
    t = np.arctan(minor / 2 * 1. / np.tan(np.radians(angle_deg)) / (major / 2))
    [max_y, min_y] = [y + minor / 2 * np.sin(t) * np.cos(np.radians(angle_deg)) +
                      major / 2 * np.cos(t) * np.sin(np.radians(angle_deg)) for t in (t, t + np.pi)]
    min_x = x-(x-min_x)/3*4
    min_y = y - (y - min_y) / 3 * 4
    max_x = x+(max_x-x)/3*4
    max_y = y+(max_y-y)/3*4
    return int(min_x), int(min_y), int(max_x), int(max_y)


dic = get_img_info()
gene_pos_img(dic)
# f = open(origin_path+file_name)
# data = f.read()
# data = data.split('\n')[2]
# print(data.split(' ')[:5])
