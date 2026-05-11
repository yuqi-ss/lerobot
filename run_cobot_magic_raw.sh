# Run script to convert dataset
# Example to run the convert_dataset_v21_to_v30.py:
# consider use video file size 500MB, data file 100MB.

python /home/yuqi/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
  --repo-id=disk-1_Arrange_the_number_7 \
  --root=/home/ss-oss1/data/user/yuqi/data/disk-1_Arrange_the_number_7 \
  --output-root=/home/ss-oss1/data/user/yuqi/data/disk-1_Arrange_the_number_7_v30/ \
  --num-workers 0 \
  --overwrite-output
  
# One ego data chunk take around 3 mins to complete.
# chunk-range Specify the chunks to process.
# repo-id: just put in a string will be fine (it is for hugging face dataset, not truly used for local dataset) 