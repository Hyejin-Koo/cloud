tsv_dir=../data
split=valid #train or valid
nshard=100
feat_dir=./MFCC_feature

for rank in $(seq 0 $((nshard - 1))); do
    python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
done
