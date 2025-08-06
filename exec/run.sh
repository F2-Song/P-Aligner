task_id=$1
src_dataset_path=$2
tgt_dataset_path=$3

cd P-Aligner
mkdir -p ../logs/$task_id/medium
python main.py \
    --id $task_id \
    --dataset_path $src_dataset_path \
    --output_path ../logs/$task_id/medium/medium.json \
    --model_path songff/P-Aligner > ../logs/$task_id/inference.log 2>&1

python parse_output.py \
    --src_dataset_path $src_dataset_path \
    --tgt_dataset_path $tgt_dataset_path \
    --medium_path ../logs/$task_id/medium/medium.json
cd ..