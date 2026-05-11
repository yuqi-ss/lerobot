# Run script to convert dataset
# Example to run the convert_dataset_v21_to_v30.py:
# consider use video file size 500MB, data file 100MB.

python /home/yuqi/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30_backup.py \
  --repo-id=database_lerobot_00 \
  --root=/home/ss-oss1/data/user/yuqi/data/database_lerobot_00 \
  --output-root=/home/ss-oss1/data/user/yuqi/data/database_lerobot_00_v30_backup/ \
  --overwrite-output
  
# One ego data chunk take around 3 mins to complete.
# chunk-range Specify the chunks to process.
# repo-id: just put in a string will be fine (it is for hugging face dataset, not truly used for local dataset) 