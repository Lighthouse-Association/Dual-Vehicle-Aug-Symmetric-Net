name='feats_aug_text'
config='feats_aug_text'

model="dual-text-cat-aggr-f"

python3 -m main --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024