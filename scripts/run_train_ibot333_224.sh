export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup torchrun --nproc_per_node=8 dinov2/train/train.py \
--config-file dinov2/configs/train/vitl16_ibot333.yaml \
--output-dir /fast/yangz16/outputs/chexfound/ibot333 \
train.dataset_path=CXRDatabase:split=TRAIN:root="/fast/yangz16/outputs/dinov2_split_512":extra="/fast/yangz16/outputs/dinov2_split_512/extra" \
&> /fast/yangz16/outputs/chexfound/ibot333.log &