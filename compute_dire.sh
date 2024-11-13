## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir

MODEL_PATH="models/lsun_bedroom.pt" # "models/lsun_bedroom.pt, models/256x256_diffusion_uncond.pt"

SAMPLE_FLAGS="--batch_size 16 --num_samples 1000  --timestep_respacing ddim20 --use_ddim True"
SAVE_FLAGS="--images_dir img_dir/imagenet_ai_0419_biggan/val/ai --recons_dir recon_dir/biggan --dire_dir dire_dir/biggan"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

DIRE/python compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder True
