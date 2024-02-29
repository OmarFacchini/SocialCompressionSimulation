#codice di esempio per i test
#--video_native e' il path ai video non condivisi su social
#--video_social e' il path ai video condivisi su social
#--video_social_emu e' il path ai video emulati social (creati da te)
#--ckpt_native path ai pesi delle 3 tecniche trainate su video non condivisi su social
#--ckpt_social path ai pesi delle 3 tecniche trainate su video condivisi su social
#--out_name path per il file di output in cui salvi i risultati

#NOTA: dovrai modificare main_largescaletest.py per eseguire il codice anche con i pesi ottenuti trainando su video emulati
#TIP: Vedi i commenti nel codice :-)
#--ckpt_social_emu emulator_scripts_OFF/emulator_scripts/dataset/emulated/Facebook/Deepfakes/libx264
python main_largescaletest.py --video_native /home/omar.facchini/SocialCompressionSimulation/emulator_scripts_OFF/emulator_scripts/dataset/nonShared/manipulated_sequences/Deepfakes/c23/videos/ --video_social /home/omar.facchini/SocialCompressionSimulation/emulator_scripts_OFF/emulator_scripts/dataset/shared/Facebook/test_facebook_deepfakes/ --video_social_emu /home/omar.facchini/SocialCompressionSimulation/emulator_scripts_OFF/emulator_scripts/dataset/emulated/Facebook/Deepfakes/libx264/ --ckpt_native /home/omar.facchini/SocialCompressionSimulation/emulator_scripts_OFF/emulator_scripts/media/SSD_new/BACKUP/home/federicomarcon/native/models/Densenet_Deepfakes.h5 --ckpt_social /home/omar.facchini/SocialCompressionSimulation/emulator_scripts_OFF/emulator_scripts/media/SSD_new/BACKUP/home/federicomarcon/facebook/models/Densenet_FB_Deepfakes.h5 --out_name /home/omar.facchini/SocialCompressionSimulation/emulator_scripts_OFF/emulator_scripts/results/Facebook/Dense_Deepfake_FB2.mat