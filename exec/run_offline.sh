export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENAI_API_BASE=http://localhost:8000/v1 # depends on vLLM
export OPENAI_API_KEY=EMPTY

task_id=$1
src_dataset_path=$2
tgt_dataset_path=$3
model_path=songff/SinglePO

cd offline_synthesis
mkdir -p ../logs/$task_id/backend
mkdir -p ../logs/$task_id/medium
mkdir -p ../logs/$task_id/synthesis

# run backend to start and serve models
bash run_serve_model.sh $model_path &
bash run_backend.sh $task_id &
sleep 180

# run synthesis process
bash run_synthesis.sh $task_id $src_dataset_path $model_path

# parse the output
cd ../online_synthesis
python parse_output_for_train.py \
    --src_dataset_path $src_dataset_path \
    --tgt_dataset_path $tgt_dataset_path \
    --tree_path ../logs/$task_id/medium/trees.json \
    --max_iterations 20

echo create $data_id data completed~!