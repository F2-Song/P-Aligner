model_path=$1

# should use the last gpu
GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
GPU_ID=$((GPU_ID - 1))

CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve $model_path \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --gpu_memory_utilization 0.9 