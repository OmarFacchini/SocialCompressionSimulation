#!/bin/bash

# dir_BASE="/media/SSD_new/BACKUP/home/federicomarcon/emu_social_v2" #path in cui vengono salvati i video

dir_BASE="dataset/emulated/" #path in cui vengono salvati i video

# my path
dir_ORIGINAL_NON_SHARED="dataset/nonShared/manipulated_sequences/" 

socials=(
  "Facebook"
  "Youtube"
)

codec=(
  "libx264"
)

main_folders=(
  "FaceShifter"
  "Deepfakes"
  #"Face2Face"
  "FaceSwap"
  #"NeuralTextures"
)

# average oneshot(using crf_2.csv):
# yt = 24.55
# fb = 27.82 
yt_crf=25 #i tuoi valori
fb_crf=28 #i tuoi valori


h264_codec="libx264" #al momento stiamo usando questo codec. gli altri dovrebbero preservare meglio al qualita' del video
vp9_codec="libvpx-vp9"
av1_codec="libaom-av1"

image_name="%04d.png"

# Creation Folders
for folder in "${socials[@]}"; do
  for codecs in "${codec[@]}"; do
    for mainf in "${main_folders[@]}"; do
      FILES="${dir_ORIGINAL_NON_SHARED}""${mainf}"/"c23/videos/*.mp4"
      for f in $FILES
        do
        echo "${dir_BASE}""${folder}"/"${mainf}"/"${codecs}"/
        mkdir -p "${dir_BASE}""${folder}"/"${mainf}"/"${codecs}"/ #crea sub-folder in path di output
        #loop video in native path
        echo "$f"
        echo "$(basename $f)"
        if [ "$folder" = "Youtube" ];
          then
              crf=$yt_crf
        else
              crf=$fb_crf
        fi
        ffmpeg -loglevel error -nostdin -i "$f" -crf "${crf}" -vcodec "${codecs}" -pix_fmt yuv420p "${dir_BASE}""${folder}"/"${mainf}"/"${codecs}"/"$(basename $f)"
      done
    done
  done
done

