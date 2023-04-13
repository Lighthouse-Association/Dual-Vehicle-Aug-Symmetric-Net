name='feats_aug_eng'
config='feats_aug_eng'

model="dual-text-cat-aggr-f"

python3 -m Track2.main --name ${name} \
--config ${config} \
--resume \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024