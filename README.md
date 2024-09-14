# Single object tracking with Siamese network
## Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)

Scripts to prepare training dataset are listed in `training_dataset` directory.


## Download pretrained backbones
Download pretrained backbones from [google driver](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) or [BaiduYun](https://pan.baidu.com/s/1pYe73PjkQx4Ph9cd3ePfCQ) (code: 5o1d) and put them into `pretrained_models` directory.

### training got10k model
cd experiments/train/got10k 
### training general model
cd experiments/train/fulldata  

CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../../tools/train.py --cfg config.yaml


# Acknowledgement
The code is implemented based on [RBO](https://github.com/sansanfree/RBO). We would like to express our sincere thanks to the contributor.


