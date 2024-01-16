import cv2
import dlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.models import load_model
import argparse
import glob
import scipy.io
import tensorflow as tf

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

def analyze_video_multiple_models(video_dir, model1, model2):#, model3):
    cap = cv2.VideoCapture(video_dir)

    count = 0  # count frames read
    numb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames
    prediction1 = []
    prediction2 = []
    #prediction3 = []

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
            prediction1.append(model1.predict(preprocessed_input)[0])
            prediction2.append(model2.predict(preprocessed_input)[0])
            #prediction3.append(model3.predict(preprocessed_input)[0])
        else:
            cap.set(1, count + 1)

        count += 1

        if (count >= numb):  # stop when all frames are processed
            break
            pbar.close()
            cap.release()
            cv2.destroyAllWindows()

    return np.asarray(prediction1), np.asarray(prediction2)#, np.asarray(prediction3)


def decode_prediction(prediction):
    class_names = ['real', 'fake']
    results = [[],[]]
    f_iter = 0
    t_iter = 0
    for i, logits in enumerate(prediction):
        breakpoint()
        class_idx = tf.argmax(logits).numpy()
        p = logits[class_idx]
        name = class_names[class_idx]

        if name == 'fake':
            results[0].append(p)
            f_iter += 1
        else:
            results[1].append(p)
            t_iter += 1
    if len(results[0]) == 0:
        print('Mean fake probability: ', 0, 'for n. frames: ', f_iter)
    else:
        print('Mean fake probability: ', np.mean(results[0]), 'for n. frames: ', f_iter)
    if len(results[0]) == 0:
        print('Mean true probability: ', 0, 'for n. frames: ', t_iter)
    else:
        print('Mean true probability: ', np.mean(results[1]), 'for n. frames: ', t_iter)
    return results, t_iter, f_iter
        #print("Frame {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))



parser = argparse.ArgumentParser()
parser.add_argument('--video_native', default='video_native', help="path to native video")
parser.add_argument('--video_social', default='video_social', help="path to social video")
parser.add_argument('--video_social_emu', default='video_social_emu', help="path to social emu video")
parser.add_argument('--ckpt_native', default='ckpt_native', help="path to ckpt native")
parser.add_argument('--ckpt_social', default='ckpt_social', help="path to ckpt social")
#parser.add_argument('--ckpt_social_emu', default='ckpt_social', help="path to ckpt social")
parser.add_argument('--out_name', default='ckpt_social', help="path to ckpt social")

args = parser.parse_args()


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for gpu_instance in physical_devices:
    print(gpu_instance)
    tf.config.experimental.set_memory_growth(gpu_instance, True)
with tf.device('/GPU:0'):
    model1 = load_model(args.ckpt_native) 
    model2 = load_model(args.ckpt_social)
    #model3 = load_model(args.ckpt_social_emu)

    videos = sorted(glob.glob(args.video_social + '*.mp4'))
    breakpoint()
    print(videos)
    results_pesi_nonsocial = [[[],[]],[[],[]],[[],[]]]
    results_pesi_social = [[[],[]],[[],[]],[[],[]]]
    #results_pesi_social_emu = [[[],[]],[[],[]],[[],[]]]


    for v in videos:
        video_native1 = args.video_native + v[len(args.video_native):]
        predictions1, predictions2 = analyze_video_multiple_models(video_native1, model1, model2)#, model3)
        result, _, _ = decode_prediction(predictions1)
        for r in result[0]:
            results_pesi_nonsocial[0][0].append(r)
        for r in result[1]:
            results_pesi_nonsocial[0][1].append(r)
        result, _, _ = decode_prediction(predictions2)
        for r in result[0]:
            results_pesi_social[0][0].append(r)
        for r in result[1]:
            results_pesi_social[0][1].append(r)
        #result, _, _ = decode_prediction(predictions3)
        #for r in result[0]:
        #    results_pesi_social_emu[0][0].append(r)
        #for r in result[1]:
        #    results_pesi_social_emu[0][1].append(r)

        ## ANALYZE VIDEO social (eg. video manipulated with NeuralTextures)

        video_social1 = args.video_social + v[len(args.video_social):]
        predictions1, predictions2 = analyze_video_multiple_models(video_social1, model1, model2)#, model3)
        result, _, _ = decode_prediction(predictions1)
        for r in result[0]:
            results_pesi_nonsocial[1][0].append(r)
        for r in result[1]:
            results_pesi_nonsocial[1][1].append(r)
        result, _, _ = decode_prediction(predictions2)
        for r in result[0]:
            results_pesi_social[1][0].append(r)
        for r in result[1]:
            results_pesi_social[1][1].append(r)
        #result, _, _ = decode_prediction(predictions3)
        #for r in result[0]:
        #    results_pesi_social_emu[1][0].append(r)
        #for r in result[1]:
        #    results_pesi_social_emu[1][1].append(r)


        ## ANALYZE VIDEO social emulated (eg. video manipulated with NeuralTextures)
        video_social_emu1 = args.video_social_emu + v[len(args.video_social_emu):]

        predictions1, predictions2, predictions3 = analyze_video_multiple_models(video_social_emu1, model1, model2)#, model3)  
        result, _, _ = decode_prediction(predictions1)
        for r in result[0]:
            results_pesi_nonsocial[2][0].append(r)
        for r in result[1]:
            results_pesi_nonsocial[2][1].append(r)
        result, _, _ = decode_prediction(predictions2)
        for r in result[0]:
            results_pesi_social[2][0].append(r)
        for r in result[1]:
            results_pesi_social[2][1].append(r)
        #result, _, _ = decode_prediction(predictions3)
        #for r in result[0]:
        #    results_pesi_social_emu[2][0].append(r)
        #for r in result[1]:
        #    results_pesi_social_emu[2][1].append(r)



    mdic = {"nsocial": results_pesi_nonsocial, "social": results_pesi_social}#, "emu_social": results_pesi_social_emu}
    scores_outpath = args.out_name
    scipy.io.savemat(scores_outpath, mdic)
