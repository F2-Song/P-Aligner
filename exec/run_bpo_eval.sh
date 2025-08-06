src_dataset_path=$1
your_model_outputs_path=$2
result_path=$3

cd utils
# conduct the comparison
python test_in_bpo.py \
    --input_file_a $src_dataset_path \
    --input_file_b $your_model_outputs_path \
    --output_file $result_path \
    --mode compare

# get the evaluation result
python test_in_bpo.py \
    --input_file_a $src_dataset_path \
    --input_file_b $your_model_outputs_path \
    --output_file $result_path \
    --mode check
