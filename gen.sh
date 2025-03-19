source ~/.bashrc

model="Qwen/Qwen2.5-Math-7B"
output_dir="./eval_result/gen"
DATA_NAME="math500,minerva_math,olympiadbench,aime24,amc23"
#minerva_math
mkdir eval_result
num_of_responses=16
my_world_size=8
NUM_GPUS=$my_world_size


for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python ./gen_data.py \
        --model_name_or_path $model \
        --dataset_name_or_path $DATA_NAME \
        --output_dir $output_dir \
        --K $num_of_responses \
        --temperature 1.0 \
        --local_index $i \
        --my_world_size $my_world_size &
done

wait # Ensure all inference processes finish

# Merge the generated data
python ./merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_merged_data.jsonl" --num_datasets ${my_world_size}

# Perform reward labeling
#python reward_labeling.py --dataset_name_or_path "${output_dir}_merged_data.jsonl" --output_dir "${output_dir}_merged_data_with_rewards.jsonl"

