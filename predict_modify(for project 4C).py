from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
import tensorflow as tf
import cv2
import time
import glob
from PIL import Image
from gtts import gTTS
import playsound
import os
import threading

def load_data(dirname):
    if dirname[-1]!='/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            # print(textname)
            with open(textname, mode = 'r', encoding="utf-8") as t:
                numbers = [float(num) for num in t.read().split()]
                # print(len(numbers))
                for i in range(len(numbers),25200):
                    numbers.extend([0.000])
            landmark_frame=[]
            row=0
            for i in range(0,70):
                landmark_frame.extend(numbers[row:row+84])
                row += 84
            landmark_frame=np.array(landmark_frame)
            # print(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,84)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    # print(Y)
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y


def load_label():
    listfile=[]
    with open("label.txt",mode='r', encoding="utf-8") as l:
        listfile=[i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label

def opencap():
    files_test = glob.glob('/Users/user/mediapipe/test/test/*.mp4')
    for filetest in files_test:
        os.remove(filetest)
    files_testout = glob.glob('/Users/user/mediapipe/testout/_test/*.mp4')
    for fileout in files_testout:
        os.remove(fileout)
    num = 1
    recOn = 1
    word = ""
    word2 = "Take w for start/stop"
    output_data_path = '/Users/user/mediapipe/test/'
    cap = cv2.VideoCapture(0)
    if not(os.path.isdir(output_data_path+"test/")):
        os.mkdir(output_data_path+"test/")
    while True:
        ret, frame_read = cap.read()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # frame_read = cv2.flip(frame_read, 1)

        # hsv = cv2.cvtColor(frame_read, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
        # mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # blurred_frame = cv2.GaussianBlur(frame_read, (25, 25), 0)
        # frame_read = np.where(mask_3d == (255, 255, 255), frame_read, blurred_frame)
        frame_show = cv2.flip(frame_read, 1)

        if recOn == 0:
            # แก้ path โฟเดอของ class ที่จะเซฟ
            out = cv2.VideoWriter("/Users/user/mediapipe/test/test/test%d.mp4" % (num), cv2.VideoWriter_fourcc(*'MP4V'), 25.0,(frame_width, frame_height))
            recOn = 2
            num += 1
        if recOn == 2:
            out.write(frame_read)
        text = word
        text2 = word2
        y0,dy = 30,30
        cv2.putText(frame_show, text, (380, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        for i, line in enumerate(text2.split('\n')):
            y = y0 + i*dy
            cv2.putText(frame_show, line, (10, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('Demo', frame_show)
        k = cv2.waitKey(1)
        if (k == ord('w')) and (recOn == 1):
            recOn = 0
            print('start')
            word = "START"
        if (k == ord('w')) and (recOn == 2):
            recOn = 1
            print('stop')
            word = "STOP"
        elif k == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break
    return

def predict():
    comp = 'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C:/Users/Asus/AppData/Local/Programs/Python/Python37/python.exe" mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu'
    os.system('set GLOG_logtostderr=1')
    cmd = 'bazel-bin\mediapipe\examples\desktop\hand_tracking\hand_tracking_cpu  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt'
    input_data_path = '/Users/user/mediapipe/test/' # ไฟล์ที่ได้จากกล้อง
    output_data_path = '/Users/user/mediapipe/testout/' #ไฟล์ที่จะเอาออก
    listfile = os.listdir(input_data_path)
    if not (os.path.isdir(output_data_path + "Absolute/")):
        os.mkdir(output_data_path + "Absolute/")
    output_dir = ""
    filel = []
    for file in listfile:
        if not (os.path.isdir(input_data_path + file)):
            continue
        word = file + "/"
        fullfilename = os.listdir(input_data_path + word)
        if not (os.path.isdir(output_data_path + "_" + word)):
            os.mkdir(output_data_path + "_" + word)
        if not (os.path.isdir(output_data_path + "Absolute/" + word)):
            os.mkdir(output_data_path + "Absolute/" + word)
        # os.system(comp)
        outputfilelist = os.listdir(output_data_path + '_' + word)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue
            filel.append(mp4list)
            inputfilen = '   --input_video_path=' + input_data_path + word + mp4list
            outputfilen = '   --output_video_path=' + output_data_path + '_' + word + mp4list
            cmdret = cmd + inputfilen + outputfilen
            os.system(cmdret)
    output_dir = '/Users/user/mediapipe/testout/Absolute'
    x_test, Y = load_data(output_dir)
    new_model = tf.keras.models.load_model('model.h5')
    # new_model.summary()

    labels = load_label()

    xhat = x_test
    yhat = new_model.predict(xhat)
    print(yhat)
    pred = [np.argmax(pred) for pred in yhat]
    predictions = np.array(pred)
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    print(predictions,rev_labels)
    no_num = 0
    x_before = 0
    word_speak = []
    for no in predictions:
        if no==32:
            no_num = 1

    filel = np.array(filel)
    txtpath = output_data_path + "result.txt"
    len_pre = len(predictions)
    if len_pre >1 and no_num == 1:
        # np.where ได้ (array([2], dtype=int64),) [0]ตัวแรกเพื่อเลือกarray [0] ตัว2 เพื่อเลือกเลขในarray
        x = (np.where(predictions == 32)[0][0])
        x_before = x - 1
    with open(txtpath, "w", encoding="utf-8") as f:
        for i in predictions:
            if no_num == 1:
                if i == predictions[x] and len_pre >1:
                    w = rev_labels[predictions[x_before]]
                    f.write(w)
                elif i == predictions[x_before] and len_pre >1:
                    w = rev_labels[predictions[x]]
                    f.write(w)
                else:
                    w = rev_labels[i]
                    f.write(w)

            elif no_num == 0:
                w = rev_labels[i]
                f.write(w)
            f.write(" ")
            word_speak.append(w)
            # if rev_labels[i] != "no" or ((rev_labels[i] == "no") and (len_pre ==1)):
            #     text = str(word_speak)
            #     tts = gTTS(text=text, lang='th')
            #     tts.save('sound.mp3')
            #     playsound.playsound('sound.mp3', True)
            #     os.system('rm sound.mp3')
                # open_gif(rev_labels[i])
            files_txt = glob.glob('/Users/user/mediapipe/testout/Absolute/test/*.txt')
            for filetxt in files_txt:
                os.remove(filetxt)
    return word_speak

def gtts():
    global word_speak
    text = str(word_speak)
    tts = gTTS(text=text, lang='th')
    tts.save('sound.mp3')
    playsound.playsound('sound.mp3', True)
    os.system('rm sound.mp3')

def anime():
    global word_speak

    for check in word_speak:
        if check == "อะไร" or check == 'เท่าไหร่' or check == 'ไหม' or check == "สวัสดี" or check == "ไม่":
            word_speak = []
            word_speak.append(check)
            break
    for anime in word_speak:
        # print(word_speak)
        if anime == "อะไร" or anime == 'เท่าไหร่' or anime == 'ไหม':
            anime = "คำถาม"
        elif anime == "สวัสดี":
            anime = "สวัสดี"
        elif anime == "ไม่":
            anime = "ไม่"
        try:
            print(anime)
            cap = cv2.VideoCapture("%s.mp4" % (anime))
            while True:
                ret, frame_read = cap.read()
                # text = "%s" % (anime)
                # win_text = text.decode('UTF-8')
                # win_text = win_text + unichr(0x0020) + unichr(0x0020)
                # Font1 = ImageFont.truetype("angsau_0.ttf", 14)
                # cv2.putText(frame_read, win_text, (70, 70), Font1, 2, (255, 0, 0), 2)
                cv2.imshow('Demo', frame_read)
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass

t1 = threading.Thread(target=gtts)
t2 = threading.Thread(target=anime)

def restart_program():
    os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == '__main__':
    opencap()
    word_speak = predict()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    #restart_program()
