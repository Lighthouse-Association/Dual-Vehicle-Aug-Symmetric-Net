name='submit'
config='feats_aug_eng'

#CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m prepare_outputs --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
TEST.BATCH_SIZE 128 \
