DATA_ROOT=../datasets
outdir=${DATA_ROOT}/exprs_map/test/

flag="--root_dir ${DATA_ROOT}
      --img_root /path/to/images
      --split MapGPT_72_scenes_processed
      --end 10  # the number of cases to be tested
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm gpt-4-vision-preview
      --response_format str
      --max_tokens 1000
      "

python vln/main_gpt.py $flag
