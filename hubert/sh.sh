python ../../fairseq_cli/hydra_train.py \
          --config-dir config/pretrain \
          --config-name hubert_base_librispeech \
          task.data=data task.label_dir=../examples/hubert/simple_kmeans/label42891 model.label_rate=100 \
          task.labels=[km]
