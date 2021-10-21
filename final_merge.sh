split=valid #train or valid
nshard=500
lab_dir=/home/nas3/user/koo/hubert/libri_label

for rank in $(seq 0 $((nshard - 1))); do
    cat ${lab_dir}/${split}_${rank}_${nshard}.km
done > ${lab_dir}/${split}.km
