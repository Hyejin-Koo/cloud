km_path=valid.km #km_path or km_path_valid
split=valid #train or valid
nshard=500
feat_dir=/home/nas3/user/koo/hubert/MFCC_feature
lab_dir=/home/nas3/user/koo/hubert/libri_label

for rank in $(seq 0 $((nshard - 1))); do
    python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
done
