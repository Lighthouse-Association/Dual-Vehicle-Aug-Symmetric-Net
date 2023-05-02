name='dual_baseline_aug3'
config='dual_baseline_aug3'

model="dual-text-cat"

python3 -m main --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024