name='eval_only'
config='dual_baseline_aug1'

python3 main.py --name ${name} \
--config ${config} \
--eval_only \
--logs-dir logs/eval/${name} \
--resume \

