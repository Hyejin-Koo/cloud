km_path=valid499_percent01.km #km_path or km_path_valid
split=valid #train or valid
nshard=100
feat_dir=./MFCC_feature
lab_dir=./label42891

for rank in $(seq 0 $((nshard - 1))); do
    python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
done
