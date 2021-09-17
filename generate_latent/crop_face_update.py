#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Dave Zhou
@LastEditors: Dave Zhou
@Description: This is used to crop facial images according to the landmarks of face.
@Date: 2019-03-07 18:25:02
@LastEditTime: 2019-03-11 15:55:35
'''


import os
from PIL import Image
import numpy as np
import dlib
from urllib.request import urlopen
import bz2
from multiprocessing import Pool
from tqdm import tqdm
import time

def getface(rgbImg):
    # print (rgbImg.mode)
    facefound = True
    detector=dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    #img = io.imread('1.jpg')
    faces = detector(rgbImg, 1)
    if len(faces) > 0:
        face=max(faces, key=lambda rect: rect.width() * rect.height())
        [x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
        fshape = landmark_predictor(rgbImg,face)
    else:
        x1=x2=y1=y2=0
        fshape = 0
        facefound = False
    return [facefound,x1,x2,y1,y2,fshape]
    

def work(imname):
    filename = imname.split('/')[-1]
    path_post = './datasets/celebA_post1/'
    w_max = 128
    h_max = 128
    post_dir = path_post+imname.split('/')[-2]
    filename = filename.split('.')[0] + '.png'
    postprocess_name = os.path.join(post_dir,filename)
    if os.path.exists(postprocess_name):
        print("File exists! Skipping image:{}".format(imname))
    else:
        if not os.path.exists(post_dir):
            os.makedirs(post_dir)
        # print ("Processing image:{}".format(os.path.join(root,imname)))
        im_tmp = Image.open(imname)
        (x,y) = im_tmp.size
        im_tmp = im_tmp.convert('RGB')
        im = np.array(im_tmp)
        
        facefound,x1,x2,y1,y2,fshape=getface(im) # get landmarks and face locations
        if facefound == False:
            print("No faces found in image{}.".format(imname))
        else:
            landmarks = np.array([[fshape.part(i).x,fshape.part(i).y] for i in range(68)])
            #print (landmarks)
            eye_dis = int(np.linalg.norm(landmarks[42] - landmarks[39])/2)
            pad_dis = 2 * (landmarks[33] - landmarks[27])
            landmarks[19] = landmarks[19]-pad_dis
            landmarks[24] = landmarks[24]+pad_dis
            # print ("eye_dis:{}, pad_dis:{}".format(eye_dis,pad_dis))
            left = min(landmarks[...,1])-eye_dis
            right = max(landmarks[...,1])+eye_dis
            up = min(landmarks[...,0])-eye_dis
            down = max(landmarks[...,0])+eye_dis
            #print(im.shape)
            rect = im[max(0,left):right,max(0,up):down]
            #print (rect.shape)
            rect = Image.fromarray(rect)

            if rect.size[0]!=w_max and rect.size[1]!=h_max:
                # rect = rect / 255.0
                rect = rect.resize((w_max, h_max), Image.BICUBIC)
            rect.save(postprocess_name)
            print ("Saving image as:{}".format(postprocess_name))

def work1(im):
    print('Processor {}'.format(im))
    print(im)    

if __name__ == '__main__':
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        f = urlopen('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as code:     
            code.write(f.read())
        t = bz2.open('shape_predictor_68_face_landmarks.dat.bz2')
        with open('shape_predictor_68_face_landmarks.dat', 'wb') as code:
            code.write(t.read())
        t.close()
        os.remove('shape_predictor_68_face_landmarks.dat.bz2')
    pathname = './datasets/celebA_post/'
    # path_post = argv[2]
    
    filelist = []
    pool = Pool(processes=4)

    for root, dirs, filenames in os.walk(pathname):
        print('Process images in dir: {}'.format(root))
        for filename in filenames:
            if filename.split('.')[-1] in ['jpg', 'png']:
                filelist.append(os.path.join(root, filename))
                # work(filelist[-1])
        for fname in tqdm(filelist):
            pool.apply_async(work, (fname, ))
    pool.close()
    pool.join()