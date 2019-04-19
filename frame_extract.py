import os
import math
import cv2
import pandas
import glob
import numpy as np


def video_to_frame():
    NoneType = type(None)
    data = pandas.read_csv("dataset/all_data.csv")
    count = 1
    nvi = vi = 0
    for index, row in data.iterrows():
        entry = row["simplified video title"]
        tag = row["tag"]
        v_list = glob.glob("dataset/videos/*_" + entry + "*.avi")
        video = cv2.VideoCapture(v_list[0])
        print(video.isOpened())
        '''
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        '''
        framerate = video.get(5)
        '''
        id = 0
        if(tag=='non-violence'):
            nvi = nvi+1
            id = nvi
        else:
            vi = vi+1
            id = vi

        if(len(glob.glob('dataset/'+str(tag))) == 0):
            os.makedirs('dataset/'+str(tag))

        o_dir = 'dataset/'+str(tag)+'/'+str(entry)+'_v'+str(id)+'.avi'
        out = cv2.VideoWriter(o_dir,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        list = glob.glob("dataset/frames/"+str(tag)+"/"+str(entry)+"*")
        if(len(list)==0):
        '''
        if(len(glob.glob('dataset/frames/'+str(tag)+'/'+str(entry)+'/')) == 0):
            os.makedirs("dataset/frames/" +str(tag)+"/"+str(entry)+"/")
        while (video.isOpened()):
            frameId = video.get(1)
            success,image = video.read()
            #img = image
            if( type(image) != NoneType):
                image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
            if (success != True):
                break
            else:
                filename = "dataset/frames/"+str(tag)+"/"+str(entry)+"/"+str(entry)+"_image_%.3d.jpg"%(frameId+1)
                print(filename)
                #out.write(img)
                cv2.imwrite(filename,image)

        video.release()
        #out.release()
        print('done')
        count+=1


if __name__ == "__main__":
    video_to_frame()
