# Run script to convert dataset
# Example to run the convert_dataset_v21_to_v30.py:
# consider use video file size 500MB, data file 100MB.

DATASET=database_lerobot_04

python /home/yuqi/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
  --repo-id=${DATASET} \
  --root=/home/ss-oss1/data/dataset/egocentric/training_egocentric/retarget_lerobot/${DATASET} \
  --output-root=/home/ss-oss1/data/user/yuqi/data/${DATASET}_v30/ \
  --num-workers 0 \
  --overwrite-output
  
# One ego data chunk take around 3 mins to complete.
# chunk-range Specify the chunks to process.
# repo-id: just put in a string will be fine (it is for hugging face dataset, not truly used for local dataset) 