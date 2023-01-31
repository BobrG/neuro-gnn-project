PROJDIR='/home/src/neuro-gnn-project/'
source activate neuro

seeds=(112078 781 912) # 121986 781 912 28119)

for s in "${seeds[@]}"; do
    echo 'seed' $s
    python3 $PROJDIR/IBGNN_modified/main_explainer.py --seed $s --cross_val --repeat 30 --explain --mask_tunning --dataset_path=$PROJDIR/datasets/ --dataset_name=Schiza --model_logger --api_key 0vu228lC9c6BKOk1wjLtyOBVz --project_name neuroml-gnn-project --experiment_name repeat_30_default_seed_$s
    echo 'renaming result log'
    mv ./logs/result.log ./logs/result_seed$s.log

done
    