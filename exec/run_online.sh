export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENAI_API_BASE=xxx
export OPENAI_API_KEY=xxx

task_id=$1
src_dataset_path=$2
tgt_dataset_path=$3

cd online_synthesis
mkdir -p ../logs/$task_id/backend
mkdir -p ../logs/$task_id/medium
mkdir -p ../logs/$task_id/synthesis

# run backend to start and serve models
bash run_backend.sh $task_id &
sleep 180

# run synthesis process
bash run_synthesis.sh $task_id $src_dataset_path

# parse the output for training P-Aligner
python parse_output_for_train.py \
    --src_dataset_path $src_dataset_path \
    --tgt_dataset_path $tgt_dataset_path \
    --tree_path ../logs/$task_id/medium/trees.json \
    --max_iterations 20

# parse the output for training SinglePO
# python parse_output_for_train_step.py \
#     --src_dataset_path $src_dataset_path \
#     --tgt_dataset_path $tgt_dataset_path \
#     --tree_path ../logs/$task_id/medium/trees.json \
#     --max_iterations 20

echo create $task_id data completed~!