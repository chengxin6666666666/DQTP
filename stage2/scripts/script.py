import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
    os.system("python train.py \
		--dataroot ./final_dataset \
		--no_dropout \
		--name EQTP3\
		--model single \
	    --self_attention \
		--dataset_mode unaligned \
	    --continue_train\
		--which_model_netG sid_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
	    --identity 1\
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
		--fineSize 256 \
	   --continue_train\
        --patchSize 32 \
		--skip 1 \
	    --no_flip \
		--batchSize 4\
		--use_norm 1 \
        --hybrid_loss \
        --times_residual \
	    --self_attention \
		--instance_norm 0 \
	    --resize_or_crop='no'\
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0 \
		--display_port=" + opt.port)

elif opt.predict:
    for i in range(1):
        os.system("python predict.py \
	        	--dataroot ./final_dataset \
	        	--name \
	        	--model single \
			    --self_attention \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
			    --no_flip \
	        	--use_norm 1 \
                --times_residual \
	        	--instance_norm 0 --resize_or_crop='no'\
	        	--which_epoch " + str(20 - i * 5))
