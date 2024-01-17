#!/bin/bash

# function that rounds a decimal number to nearest int
round() {
    echo "scale=0; ($1 + 0.5) / 1" | bc
}

# my path
dir_BASE="../../../../../../media/mmlab/Datasets_4TB/ff++_shared/FaceForensics++_shared/"

# original path
# dir_BASE="ff++_shared/FaceForensics++_shared/"  #modifica con tuo path video social shared

# my path
dir_IMMAGINI="../../../../../../media/mmlab/Datasets_4TB/FaceForensics++/manipulated_sequences/" 

# original path
# dir_IMMAGINI="/media/SSD_new/FaceForensics++/manipulated_sequences/"  #modifica con tuo path video non shared

# should use the 2 path above, in row 83-84 and 86-87?

# socials used
socials=(
  "Facebook"
  "YouTube"
)

#codec used
codec=(
  "libx264"
)

# folders of FaceForencics manipulation techniques
# chosen 3/5 techniques as requested
main_folders=(
  "FaceShifter"
  "Deepfakes"
  #"Face2Face"
  "FaceSwap"
  #"NeuralTextures"
)

# 0000.png
image_name="%04d.png"

tau=1

# stores one-shot crf
array1=()

# stores crf values tested, in this case 20-51
array2=()

# stores duration
array3=()

crf_1_file="crf_1.csv"
crf_2_file="crf_2.csv"
duration_file="duration.csv"

# check existence of the files, if exist -> remove
if [ -e "test.mp4" ]; then
    rm -f "test.mp4"
    echo "File test.mp4 removed."
fi
if [ -e "$crf_1_file" ]; then
    rm -f "$crf_1_file"
    echo "File $crf_1_file removed."
fi
if [ -e "$crf_2_file" ]; then
    rm -f "$crf_2_file"
    echo "File $crf_2_file removed."
fi
if [ -e "$duration_file" ]; then
    rm -f "$duration_file"
    echo "File $duration_file removed."
fi

# combine socials, techniques and codecs 
for folder in "${socials[@]}"; do
  crf_1_file="${folder}"/"crf_1.csv"
  crf_2_file="${folder}"/"crf_2.csv"
  duration_file="${folder}"/"duration.csv"
  arr_bitstream=0
  iter=0
  for codecs in "${codec[@]}"; do
    for mainf in "${main_folders[@]}"; do
      if [ "$folder" = "YouTube" ];
      then
        # get path of videos for the social and technique used

        #my values
        FILES="${dir_BASE}""${folder}"/"val_youtube"_"${mainf}"/"*.mp4"
        BASE_PATH="${dir_IMMAGINI}""${mainf}""/c23/videos"

        # original values
        # FILES="ff++_shared/FaceForensics++_shared"/"${folder}"/"val_youtube"_"${mainf}"/"*.mp4"
        # BASE_PATH="/media/SSD_new/FaceForensics++/manipulated_sequences/""${mainf}""/c23/videos"
      else

        #my values
        FILES="${dir_BASE}""${folder}"/"val_facebook"_"${mainf}"/"*.mp4"
        BASE_PATH="${dir_IMMAGINI}""${mainf}""/c23/videos"

        # original values
        # FILES="ff++_shared/FaceForensics++_shared"/"${folder}"/"val_facebook"_"${mainf}"/"*.mp4"
        # BASE_PATH="/media/SSD_new/FaceForensics++/manipulated_sequences/""${mainf}""/c23/videos"
      fi
      # calculate all informations needed on the file usig ffprobe
      for f in $FILES
          do
          frame_ratio=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "$f")
          pix_fmt=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=pix_fmt "$f")
          post_size=$(ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 "$f")
          ori_size=$(ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 $BASE_PATH/"$(basename $f)")
          duration=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=duration $BASE_PATH/"$(basename $f)")
          array3+=("$duration")

          crf_needed=$(echo "23 + l($post_size/$ori_size)/l(1-0.1285)" | bc -l) 
          array1+=( "$crf_needed" )
          echo "---vvv---"
          echo $ori_size
          echo $post_size
          echo $crf_needed
          echo "---iii---"
      
          crt=20
          flag_search=1
          echo "$f"
          echo $BASE_PATH/"$(basename $f)"
          # test different compression rates (20 to 51)
          for i in $(seq 20 51); do  
            ffmpeg -loglevel error -nostdin -i $BASE_PATH/"$(basename $f)" -r "${frame_ratio}" -crf $i -vcodec "${codecs}" -pix_fmt "${pix_fmt}" "test.mp4"
            d=$(ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 test.mp4)
            e=$(ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 $f)
            d=$(echo "$d" | sed 's:.*=::')
	          e=$(echo "$e" | sed 's:.*=::') 
            rm test.mp4
            if [ "$d" -lt "$e" ]; then
                  crt=$((i))
                  array2+=( "$crt" )
                  echo $crt
                  break
            fi
          done
          

	        arr_bitstream=$((arr_bitstream+crt))
          iter=$((iter+1))
          echo "######mean CRT#######"
          result=$(echo "scale=2; $arr_bitstream / $iter" | bc)
          echo $result
          echo "#####################"
      done
    done
  done
  for element in "${array1[@]}"; do
    echo "$element" >> "$crf_1_file"
  done

  for element in "${array2[@]}"; do
    echo "$element" >> "$crf_2_file"
  done

  for element in "${array3[@]}"; do
    echo "$element" >> "$duration_file"
  done
  echo $((arr_bitstream/iter)) > "${folder}".txt
done

