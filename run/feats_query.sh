name='feats_query'
config='feats_query'

model="dual-text-cat-aggr-f"

python3 -m Track2.main --name ${name} \
--config ${config} \
--logs-dir logs/${name} \
MODEL.NAME ${model} \
MODEL.SAME_TEXT True \
MODEL.MERGE_DIM 1024