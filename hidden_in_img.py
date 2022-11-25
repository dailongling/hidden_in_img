# -*- coding: utf-8-*-
# Author: Jack Cui
import sys
from turtle import width
import cv2
from PIL import Image
import numpy as np

def parse_from_img(img_path):
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape

    stepx = 2
    stepy = 2

    sml_w = img_w // stepx
    sml_h = img_h // stepy

    res_img = np.zeros((sml_h, sml_w, 3), np.uint8)

    for m in range(0, sml_w):
        for n in range(0, sml_h):
            map_colS = int(m * stepx + stepx * 0.5-1) # 参考点
            map_rowS = int(n * stepy + stepy * 0.5-1)
            map_colR = int(m * stepx + stepx * 0.5-1)
            map_rowR = int(n * stepy + stepy * 0.5)
            map_colG = int(m * stepx + stepx * 0.5)
            map_rowG = int(n * stepy + stepy * 0.5-1)
            map_colB = int(m * stepx + stepx * 0.5)
            map_rowB = int(n * stepy + stepy * 0.5)
            # 用3个点恢复成1个点
            res_img[n, m] = three2one(ab_diff_value(img[map_rowS, map_colS], img[map_rowR, map_colR], img[map_rowG, map_colG], img[map_rowB, map_colB]))
    return res_img

def one2three(pointsmal):
    # 入参的是承载隐藏信息的点值list:[R,G,B]
    # 把一个点的RGB值平均分配到2个点，一个点承载R，一个点承载G，一个点承载B
    R_str = str(bin(pointsmal[0]))[2:].zfill(9)
    G_str = str(bin(pointsmal[1]))[2:].zfill(9) # 转成二进制9位str -> 001111101
    B_str = str(bin(pointsmal[2]))[2:].zfill(9)    
    R_point = [int(R_str[:3], 2), int(R_str[3:6], 2), int(R_str[6:], 2)]
    G_point = [int(G_str[:3], 2), int(G_str[3:6], 2), int(G_str[6:], 2)] # 将9位二进制分成3段，每段只有3位，表示RGB的其中一个值，峰值最大变成了7
    B_point = [int(B_str[:3], 2), int(B_str[3:6], 2), int(B_str[6:], 2)]
    return [R_point, G_point, B_point]

def three2one(three_point):
    # 把3个点恢复成1个点
    point0, point1, point2 = three_point
    R_str = str(bin(point0[0]))[2:].zfill(3) + str(bin(point0[1]))[2:].zfill(3) + str(bin(point0[2]))[2:].zfill(3)
    G_str = str(bin(point1[0]))[2:].zfill(3) + str(bin(point1[1]))[2:].zfill(3) + str(bin(point1[2]))[2:].zfill(3)
    B_str = str(bin(point2[0]))[2:].zfill(3) + str(bin(point2[1]))[2:].zfill(3) + str(bin(point2[2]))[2:].zfill(3)
    R = int(R_str, 2)
    G = int(G_str, 2)
    B = int(B_str, 2)
    return [R, G, B]

def deal_orign_img(img_path, gather_scale):
    # 从大图里面抽出小图，gather_scale=2的时候，相当于取了原图的1/4尺寸
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    sml_w = img_w // gather_scale
    sml_h = img_h // gather_scale
    res_img = np.zeros((sml_h, sml_w, 3), np.uint8)

    for m in range(0, sml_w):
        for n in range(0, sml_h):
            map_col = int(m * gather_scale + gather_scale * 0.5)
            map_row = int(n * gather_scale + gather_scale * 0.5)
            res_img[n, m] = img[map_row, map_col]
            res_img[n, m] = res_img[n, m]
    return res_img

def diff_value(pointbig, pointsmal):
    # 差值法，把要藏的点改成和旁边点的差值
    # 因为pointsmal里面的值不会超过16，所以如果有负值出现，那就加，相加也不会超过30
    # 解码的时候直接取差值绝对值就行
    out = []
    for i in range(3):
        delta = int(pointbig[i]) - int(pointsmal[i])
        if delta >= 0:
            out.append(delta)
        else:
            out.append(pointbig[i] + pointsmal[i])
    return out

def ab_diff_value(pointbig, pointR, pointG, pointB):
    # 解码的时候直接取差值就行
    outR = []
    outG = []
    outB = []
    for i in range(3):
        deltaR = abs(int(pointbig[i]) - int(pointR[i]))
        deltaG = abs(int(pointbig[i]) - int(pointG[i]))
        deltaB = abs(int(pointbig[i]) - int(pointB[i]))
        outR.append(deltaR)
        outG.append(deltaG)
        outB.append(deltaB)
    return [outR, outG, outB]


def generate_img(big_img_path, small_img_path):

    big_img = cv2.imread(big_img_path)
    sml_img = cv2.imread(small_img_path)

    dst_img = big_img.copy()

    big_h, big_w, _ = big_img.shape
    sml_h, sml_w, _ = sml_img.shape

    stepx = big_w / sml_w
    stepy = big_h / sml_h

    for m in range(0, sml_w):
        for n in range(0, sml_h):
            # 把sml_img 平均分配到3个点, 取4个点，左上角的点不改动。左下角存R，右上角存G，右下角存B
            point0, point1, point2 = one2three(sml_img[n, m])
            map_colR = int(m * stepx + stepx * 0.5 -1)
            map_rowR = int(n * stepy + stepy * 0.5)          
            map_colG = int(m * stepx + stepx * 0.5)
            map_rowG = int(n * stepy + stepy * 0.5 -1)
            map_colB = int(m * stepx + stepx * 0.5)
            map_rowB = int(n * stepy + stepy * 0.5)
            if map_colR < big_w and map_rowR < big_h:
                dst_img[map_rowR, map_colR] = diff_value(dst_img[map_rowR, map_colR], point0)
            if map_colG < big_w and map_rowG < big_h:
                dst_img[map_rowG, map_colG] = diff_value(dst_img[map_rowG, map_colG], point1)
            if map_colB < big_w and map_rowB < big_h:
                dst_img[map_rowB, map_colB] = diff_value(dst_img[map_rowB, map_colB], point2)
    return dst_img

def Img2Text(img_fname):
    img = cv2.imread(img_fname)
    height, width, _ = img.shape
    text_list = []
    for h in range(height):
        for w in range(width):
            R, G, B = img[h, w]
            if R | G | B == 0:
                break
            idx = (G << 8) + B
            text_list.append(chr(idx))
    text = "".join(text_list)
    with open("斗破苍穹_dec.txt", "a", encoding="utf-8") as f:
        f.write(text)

def Text2Img(txt_fname, save_fname):
    with open(txt_fname, "r", encoding="utf-8") as f:
        text = f.read()
    text_len = len(text)
    img_w = 1000
    img_h = 1680
    img = np.zeros((img_h, img_w, 3))
    x = 0
    y = 0
    for each_text in text:
        idx = ord(each_text)  # 返回对应的ascii码
        rgb = (0, (idx & 0xFF00) >> 8, idx & 0xFF)
        img[y, x] = rgb
        if x == img_w - 1:
            x = 0
            y += 1
        else:
            x += 1
    cv2.imwrite(save_fname, img)

def big_with_small(big_img_path, small_img_path, res_img_path):
    """
    大图里藏小图
    """
    dst_img = generate_img(big_img_path, small_img_path)
    cv2.imwrite(res_img_path, dst_img)

if __name__ == "__main__":
    # 处理一下原图，让原图中没有低于16的RGB值
    # deal_orign_img("1.png")
    # # 将小图藏于大图中，并保存结果
    big_img_path = "1_2k.png"
    small_img_path = "2.png"
    res_img_path = "res.png"
    # big_with_small(big_img_path, small_img_path, res_img_path)

    # # 从大图中，解析出小图
    parsed_img_fname = "parsed_img.png"
    parsed_img = parse_from_img(res_img_path)
    # cv2.imwrite(parsed_img_fname, parsed_img)

    # 文本生成图片
    txt_fname = "dpcq.txt"
    txt_img_fname = "dpcq.png"
    txt_res_img_path = "text_res.png"
    Text2Img(txt_fname, txt_img_fname)

    # 将生成的文本图片，藏于大图中，并保存结果
    big_with_small(big_img_path, txt_img_fname, txt_res_img_path)

    # 从藏有文本的大图中，解析出文本小图  
    parsed_img_text_fname = "pares_text_img.png"
    parsed_img_text = parse_from_img(txt_res_img_path)
    cv2.imwrite(parsed_img_text_fname, parsed_img_text)

    # # 从文本图片中，解析出文字
    Img2Text(parsed_img_text_fname)
