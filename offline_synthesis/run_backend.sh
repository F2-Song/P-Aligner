task_id=$1

# should be defind as CUDA_VISIBLE_DEVICES - 1, leaving one gpu for SinglePO
rank_sum=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
rank_sum=$((rank_sum - 1))
each_gpu=1

cd ../online_synthesis

touch ../logs/$task_id/thread_flag

for rank in $(seq 0 $((rank_sum - 1))); do
    start_gpu=$((rank * each_gpu))
    gpu_list=$(seq $start_gpu $((start_gpu + each_gpu - 1)))

    GPUS=$(echo $gpu_list | tr ' ' ',')

    echo "Running rank $rank with GPUs $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS python backend.py \
        --flag_path ../logs/$task_id/thread_flag \
        --lm_model_path meta-llama/Llama-3.1-8B \
        --rm_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
        --port 1200$rank > ../logs/$task_id/backend/backend_1200${rank}.log 2>&1 &
done

wait

rm ../logs/$task_id/thread_flag