import cv2
import dlib
import numpy as np
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from keras.models import load_model
import argparse



# from github FaceForensics++
def get_boundingbox(face, width, height, scale=1.3):  # takes a dlib face to return a bounding box
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    size_bb = int(max(x2 - x1, y2 - y1) * scale)  # size of bounding box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_input(cropped_frame):
    img = Image.fromarray(cropped_frame)
    img = img.resize([299, 299], Image.NEAREST)
    img = np.asarray(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# analyze the whole video
def analyze_video(video_dir, model):
    cap = cv2.VideoCapture(video_dir)

    count = 0  # count frames read
    numb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames
    prediction = []

    pbar = tqdm(total=numb - count, position=0, leave=True, desc='Analyzing video: ')

    while (cap.isOpened()):
        ret, frame = cap.read()
        pbar.update(1)

        if not ret:
            break

        height, width = frame.shape[:2]

        face_detector = dlib.get_frontal_face_detector()  # set face detector model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        faces = face_detector(gray, 1)  # detect faces

        if len(faces):  # select first largest face
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)  # get bounding box
            cropped_face = frame[y:y + size, x:x + size]  # crop face
            preprocessed_input = preprocess_input(cropped_face)  # preprocess for feeding network
            prediction.append(model.predict(preprocessed_input)[0])
        else:
            cap.set(1, count + 1)

        count += 1

        if (count >= numb):  # stop when all frames are processed
            break
            pbar.close()
            cap.release()
            cv2.destroyAllWindows()

    return np.asarray(prediction)


def decode_prediction(prediction):
    class_names = ['real', 'fake']
    results = [[],[]]
    f_iter = 0
    t_iter = 0
    for i, logits in enumerate(prediction):
        class_idx = tf.argmax(logits).numpy()
        p = logits[class_idx]
        name = class_names[class_idx]

        if name == 'fake':
            results[0].append(p)
            f_iter += 1
        else:
            results[1].append(p)
            t_iter += 1
    print('Mean fake probability: ', np.mean(results[0]), 'for n. frames: ', f_iter)
    print('Mean true probability: ', np.mean(results[1]), 'for n. frames: ', t_iter)

        #print("Frame {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))



parser = argparse.ArgumentParser()
parser.add_argument('--video_native', default='video_native', help="path to native video")
parser.add_argument('--video_social', default='video_social', help="path to social video")
parser.add_argument('--video_social_emu', default='video_social_emu', help="path to social emu video")
parser.add_argument('--ckpt_native', default='ckpt_native', help="path to ckpt native")
parser.add_argument('--ckpt_social', default='ckpt_social', help="path to ckpt social")
args = parser.parse_args()

## ANALYZE VIDEO native (eg. video manipulated with NeuralTextures)
# load model

model = load_model(args.ckpt_native) #check that you have this path, else modify it

predictions = analyze_video(args.video_native, model)
decode_prediction(predictions)

## ANALYZE VIDEO social (eg. video manipulated with NeuralTextures)

predictions = analyze_video(args.video_social, model)
decode_prediction(predictions)

## ANALYZE VIDEO social emulated (eg. video manipulated with NeuralTextures)

predictions = analyze_video(args.video_social_emu, model)
decode_prediction(predictions)

#------SOCIAL NETWORK WEIGHTS-----

model = load_model(args.ckpt_social) #check that you have this path, else modify it

predictions = analyze_video(args.video_native, model)
decode_prediction(predictions)

## ANALYZE VIDEO social (eg. video manipulated with NeuralTextures)

predictions = analyze_video(args.video_social, model)
decode_prediction(predictions)

## ANALYZE VIDEO social emulated (eg. video manipulated with NeuralTextures)

predictions = analyze_video(args.video_social_emu, model)
decode_prediction(predictions)
