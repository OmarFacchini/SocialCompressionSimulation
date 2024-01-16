# import

import numpy as np
import os
import cv2
import dlib
import random
import json
import glob

from tqdm import tqdm  # for progress bar


# function from github FaceForensics++
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


# extract every N frames in a video (N=10 in our configuration)
#extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social/Facebook/Deepfakes/frames/train')
def extract_frames(path, dest, N=10):
    # create output folder
    name, extension = os.path.splitext(path)
    output_path = os.path.join(dest, name.rsplit('/', 1)[1]); print(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

        # open video
    cap = cv2.VideoCapture(path)
    numb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get number of frames

    count = 0  # count frames read
    face_detector = dlib.get_frontal_face_detector()  # set face detector model
    pbar = tqdm(total=N - count, position=0, leave=True, desc='Extracting frames: ')  # set progress bar
    if numb > 0:
        while (count < N):
            frame_idx = random.randint(0, numb - 1)
            cap.set(1, frame_idx)  # pick a random frame
            ret, frame = cap.read()

            if not ret:
                print('Error reading frames')
                break

            height, width = frame.shape[:2]  # get frame shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            faces = face_detector(gray, 1)  # detect faces

            if len(faces):
                face = faces[0]
                x, y, size = get_boundingbox(face, width, height)  # get bounding box
                cropped_face = frame[y:y + size, x:x + size]  # crop face
                cv2.imwrite(os.path.join(output_path, '{:04d}.png'.format(frame_idx)), cropped_face)
            else:
                cap.set(1, count + 1)  # if face not detected, read next frame

            pbar.update(1)
            count = count + 1

    cap.release()
    pbar.close()
    cv2.destroyAllWindows()


# obtaining list of videos of class "original"
def get_original_filenames(json_path, dataset_path):
    with open(json_path, mode='r') as j:  # read json
        l = json.load(j)

    original_filenames = []
    for x in l:
        for y in x:
            name = dataset_path + y + '.mp4'
            original_filenames.append(name)

    return original_filenames


# obtaining list of videos of class "manipulated"
def get_manipulated_filenames(json_path, dataset_path):
    with open(json_path, mode='r') as j:
        l = json.load(j)

    manipulated_filenames = []

    for x in range(len(l)):
        name1 = dataset_path + l[x][0] + '_' + l[x][1] + '.mp4'
        manipulated_filenames.append(name1)
        name2 = dataset_path + l[x][1] + '_' + l[x][0] + '.mp4'
        manipulated_filenames.append(name2)

    return manipulated_filenames

# train folder (eg. Deepfakes videos)

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/train')
#os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/train')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/train')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/train')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/train')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/train')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/train')

##########################

# train folder (eg. Deepfakes videos)
print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/val')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/val')

for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/val')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/val')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/val')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/val')


##########################################################################
# train folder (eg. Deepfakes videos)
print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/test')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/test')

for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Face2Face/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/test')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceShifter/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/test')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/FaceSwap/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json', '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/test')
for i in tqdm(train_manipulated_filenames):
  extract_frames(i,'/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/NeuralTextures/frames/test')


##########################################################
# train folder (eg. Deepfakes videos)

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/libx264/') #Youtube
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/train')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/train')

for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/train')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/train')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/train')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('train.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/train'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/train')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/train')

##########################

# train folder (eg. Deepfakes videos)
print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/val')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/val')

for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/val')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/val')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/val')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('val.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/val'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/val')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/val')

##########################################################################
# train folder (eg. Deepfakes videos)
print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/test')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Deepfakes/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/test')

for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/Face2Face/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/test')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceShifter/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/test')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/FaceSwap/frames/test')

print('Train folder:')
train_manipulated_filenames = get_manipulated_filenames('test.json',
                                                        '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/libx264/')
# manipulated subfolder
if not os.path.exists('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/test'):
    os.mkdir('/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/test')
for i in tqdm(train_manipulated_filenames):
    extract_frames(i, '/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Youtube/NeuralTextures/frames/test')

