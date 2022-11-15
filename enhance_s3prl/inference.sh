python s3prl/run_downstream.py -m enhance_inference \
	-i result/downstream/AIhub2/stft/best-states-dev.ckpt \
	-c ~/enhancelink/configs/cfg_voicebank.yaml \
	-n AIhub2/inference \
	-u stft_mag -g 's3prl/upstream/log_stft/stft_mag.yaml'\
	-d enhancement_stft2 \
	-t inference
