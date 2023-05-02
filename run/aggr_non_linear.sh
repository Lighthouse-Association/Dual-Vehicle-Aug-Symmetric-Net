name='aggr_non_linear'
config='aggr_non_linear'

model="dual-text-cat-aggr-f"

python3 -m main --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024