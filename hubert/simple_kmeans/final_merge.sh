split=valid #train or valid
nshard=100
lab_dir=./label42891

for rank in $(seq 0 $((nshard - 1))); do
    cat ${lab_dir}/${split}_${rank}_${nshard}.km
done > ${lab_dir}/${split}.km
