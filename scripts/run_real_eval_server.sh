model_name=testing
ckpt=/run/determined/workdir/home/unifolm-world-model-action/step47200.ckpt
config=configs/inference/world_model_decision_making.yaml
seed=123
res_dir="/run/determined/workdir/home/unifolm-world-model-action/results"
datasets=(
    "metal_part_sort_v5"
    "metal_part_sort_v6"
)


for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/real_eval_server.py \
    --seed ${seed} \
    --ckpt_path $ckpt \
    --config $config \
    --savedir "${res_dir}/${dataset}/${model_name}/videos" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 16 \
    --ddim_eta 1.0 \
    --video_length 16 \
    --frame_stride 2 \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 \
    --perframe_ae
done
