#MODIFICA I PATHs
#--path path al dataset emulato
#--model path al modello pre-trainato su video non social
#--out_model path alla cartella dove salvi il modello trainato su video emulati

python train.py --path /media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2/Facebook/Deepfakes/frames --model Models/Dense/Densenet_Deepfakes.h5 --out_model Models_v2/Dense/Densenet_FB_Deepfakes.h5

