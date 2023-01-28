# miniImagenet
python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir miniImagenet_5w1s -lamb 2.0 -lamb_diff 0.5 -semantic_path /data/FSLDatasets/LPE_dataset/miniimagenet/wnid2CLIPemb_zscore.npy -sem_dim 512 -is_LPE -n_attr_templet 5 -templet_weight sem_generate
python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir miniImagenet_5w5s -lamb 2.0 -lamb_diff 1 -semantic_path /data/FSLDatasets/LPE_dataset/miniimagenet/wnid2CLIPemb_zscore.npy -sem_dim 512 -is_LPE -n_attr_templet 5 -templet_weight sem_generate -shot 5 -milestones 40 50 -max_epoch 60

#cub
python train.py -batch 64 -dataset cub -gpu 0 -extra_dir cub_5w1s -lamb 2.0 -lamb_diff 0.5 -semantic_path /data/FSLDatasets/LPE_dataset/cub/wnid2vec.npy -sem_dim 312 -is_LPE -n_attr_templet 5 -templet_weight sem_generate
python train.py -batch 64 -dataset cub -gpu 0 -extra_dir cub_5w5s -lamb 2.0 -lamb_diff 0.5 -semantic_path /data/FSLDatasets/LPE_dataset/cub/wnid2vec.npy -sem_dim 312 -is_LPE -n_attr_templet 5 -templet_weight sem_generate -shot 5 -milestones 40 50 -max_epoch 60

# cifar-fs
python train.py -batch 128 -dataset cifar_fs -gpu 0 -extra_dir cifarfs_5w1s -lamb 2.0 -lamb_diff 1 -semantic_path /data/FSLDatasets/LPE_dataset/cifar_fs/wnid2CLIPemb_zscore.npy -sem_dim 512 -is_LPE -n_attr_templet 5 -templet_weight sem_generate -milestones 40 50 -max_epoch 60
python train.py -batch 128 -dataset cifar_fs -gpu 0 -extra_dir cifarfs_5w5s -lamb 2.0 -lamb_diff 1 -semantic_path /data/FSLDatasets/LPE_dataset/cifar_fs/wnid2CLIPemb_zscore.npy -sem_dim 512 -is_LPE -n_attr_templet 5 -templet_weight sem_generate -shot 5 -milestones 40 50 -max_epoch 60

# tieredImagenet
python train.py -batch 416 -dataset tieredimagenet -gpu 0 -extra_dir tieredImagenet_5w1s -lamb 0.5 -lamb_diff 0.5 -semantic_path /data/FSLDatasets/LPE_dataset/tiered_imagenet/wnid2CLIPemb_zscore.npy -sem_dim 512 -is_LPE -n_attr_templet 5 -templet_weight sem_generate -milestones 40 50 -max_epoch 60 -no_queryatt
python train.py -batch 256 -dataset tieredimagenet -gpu 0 -extra_dir tieredImagenet_5w5s -lamb 0.5 -lamb_diff 0.5 -semantic_path /data/FSLDatasets/LPE_dataset/tiered_imagenet/wnid2CLIPemb_zscore.npy -sem_dim 512 -is_LPE -n_attr_templet 5 -templet_weight sem_generate -milestones 40 50 -max_epoch 60 -no_queryatt -shot 5


