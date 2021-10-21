tsv_dir=/home/koo/fairseq_my/examples/wav2vec/data/DCASE2021+Libriother #../data
split=train #train or valid
nshard=100
feat_dir=/home/nas3/user/koo/hubert/mfcc_feature_nshard100

for rank in $(seq 0 $((nshard - 1))); do
    python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
done
