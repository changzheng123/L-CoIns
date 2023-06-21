# L-CoIns: Language-based Colorization with Instance Awareness

## Abstract
Language-based colorization produces plausible colors consistent with the language description provided by the user. Recent studies introduce additional annotation to prevent color-object coupling and mismatch issues, but they still have difficulty in distinguishing instances corresponding to the same object words. In this paper, we propose a transformer-based framework to automatically aggregate similar image patches and achieve instance awareness without any additional knowledge. By applying our presented luminance augmentation and counter-color loss to break down the statistical correlation between luminance and color words, our model is driven to synthesize colors with better descriptive consistency. We further collect a dataset to provide distinctive visual characteristics and detailed language descriptions for multiple instances in the same image. Extensive experiments demonstrate our advantages of synthesizing visually pleasing and descriptionconsistent results of instance-aware colorization.

<img src="teaser.png" align=center />


## Prerequisites
* Python 3.6
* PyTorch 1.10
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
https://github.com/changzheng123/L-CoIns.git
```
Install PyTorch and dependencies
```
http://pytorch.org
```
Install other python requirements
```
pip install -r requirement.txt
```


## Datasets
We process the [MSCOCO](https://cocodataset.org/) dataset for evaluation. Specifically, we keep the images whose captions contain adjectives and annotate the correspondence between adjectives and nouns in the caption to produce the ground-truth object-color corresponding matrix (OCCM). Metadata is in ``L-CoDer/resources``.

Extended COCO-Stuff dataset lacks samples with distinctive visual characteristics and detailed language description or multiple instances in image (right image). Therefore, we build the new dataset [Multi-instance dataset]() with these miscellaneous cases to train the model to learn inter-instance relationships and assign distinct colors to each instance.

## Testing with pretrained model
```
python run_colorization.py --model colorization_vit_base_patch16_224_group_post --output_dir output/colorization_group_post --data_path <your data path> --finetune output/colorization_group_post/checkpoint-best.pth  --batch_size 20 --opt adamw --dist_eval --test
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [L-CoIns: Language-based Colorization with Instance Awareness]([Whttps://ci.idm.pku.edu.cn/Weng_CVPR23f.pdf)
```
@InProceedings{lcoins,
  author = {Chang, Zheng and Weng, Shuchen and Zhang, Peixuan and Li, Yu and Li, Si and Shi, Boxin},
  title = {L-CoIns: Language-based Colorization with Instance Awareness},
  booktitle = {{CVPR}},
  year = {2023}
}
```
