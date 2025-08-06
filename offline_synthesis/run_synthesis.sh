task_id=$1
src_dataset_path=$2
model_path=$3

# get available gpu num via CUDA_VISIBLE_DEVICES
rank_sum=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
rank_sum=$((rank_sum - 1))

cd ../online_synthesis

for rank in $(seq 0 $((rank_sum - 1))); do
    echo "Running MCTS of rank $rank on $task_id"
    python main.py \
        --id $task_id \
        --iterations 20 \
        --num_threads 20 \
        --exploration_weight 0.1 \
        --dataset_path $src_dataset_path \
        --last_output_path ../logs/$task_id/medium/trees.json \
        --output_path ../logs/$task_id/medium/trees.json \
        --flag_path ../logs/$task_id/thread_flag \
        --model_path $model_path \
        --lm_optimizer_type local \
        --port 1200$rank \
        --rank $rank \
        --rank_sum $rank_sum > ../logs/$task_id/synthesis/synthesis_1200${rank}.log 2>&1 &
done

wait

rm ../logs/$task_id/thread_flag
pgrep -f $task_id | xargs kill -9
pgrep -f $model_path | xargs kill -9