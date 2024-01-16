#codice di esempio per i test
#--video_native e' il path ai video non condivisi su social
#--video_social e' il path ai video condivisi su social
#--video_social_emu e' il path ai video emulati social (creati da te)
#--ckpt_native path ai pesi delle 3 tecniche trainate su video non condivisi su social
#--ckpt_social path ai pesi delle 3 tecniche trainate su video condivisi su social
#--out_name path per il file di output in cui salvi i risultati

#NOTA: dovrai modificare main_largescaletest.py per eseguire il codice anche con i pesi ottenuti trainando su video emulati
#TIP: Vedi i commenti nel codice :-)

python main_largescaletest.py --video_native /media/SSD_new/FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos/ --video_social ff++_shared/FaceForensics++_shared/YouTube/test_youtube_neuraltextures/ --video_social_emu /media/SSD_new/BACKUP/home/federicomarcon/emu_social/Youtube/NeuralTextures/libx264/ --ckpt_native Models/Dense/Densenet_NeuralTextures.h5 --ckpt_social Models/Dense/social/Densenet_YT__NeuralTextures.h5 --out_name Dense_NT_YT.mat

