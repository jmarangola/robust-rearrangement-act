#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-4:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=checkpoint_sweep

# ykosiypt, qzt2lh9x, frmymr5x, xaqzl3mx, 8k6oiit5, otft0k6k, 9zjnzg4r

run_id="ol-state-dr-med-1/9zjnzg4r"

root_dir=outputs

folder=$(dirname "$root_dir/$run_id")

if [ ! -d "$folder" ]; then
    echo "Folder '$folder' does not exist. Creating it..."
    mkdir -p "$folder"
fi

csv_file="$root_dir/${run_id}_results.csv"
echo "wt_type,success_rate" > "$csv_file"

for ((i=99; i<=4999; i+=100)); do
    wt_type="_$i.pt"
    
    output=$(python -m src.eval.evaluate_model --run-id "$run_id" --n-envs 128 --n-rollouts 128 -f one_leg --if-exists append --max-rollout-steps 700 --controller diffik --use-new-env --action-type pos --observation-space state --randomness med --wt-type "$wt_type")
    
    success_rate=$(echo "$output" | grep -oP "Success rate: \K[\d.]+")
    success_count=$(echo "$output" | grep -oP "Success rate: [\d.]+% \(\K\d+")
    rollout_count=$(echo "$output" | grep -oP "Success rate: [\d.]+% \(\d+/\K\d+")
    
    echo "$i,$success_rate" >> "$csv_file"
    echo "wt_type: $wt_type, Success rate: $success_rate ($success_count/$rollout_count)"
done