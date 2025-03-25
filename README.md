<div align="center">

# Cross-View Geo-Localization with Panoramic Street-View and VHR Satellite Imagery in Decentrality Settings
[![Paper](http://img.shields.io/badge/paper-arXiv.2412.11529-B31B1B.svg)](https://arxiv.org/abs/2412.11529)



## To do list

1. Tools for preprocessing Street View panoramas and establishing custom datasets.





## 1. Decentrality description

The cross-view geo-localization (CVGL) task aims to retrieve the best-matched satellite reference image from a database for each query street-view image. Traditionally, datasets such as CVUSA and CVACT adopt a center-aligned organization, ensuring that each query is precisely centered within its corresponding paired reference image. However, this idealized setting does not reflect real-world applications, where reference images are typically pre-collected **without explicit alignment** to queries.

To construct a more practical reference database, seamless tiling methods (e.g., VIGOR) are employed. However, this inevitably introduces **positional offsets**, which we define as **decentrality**, where queries are not always located at the center of their best-matched reference images. As decentrality increases, queries tend to be positioned toward the edges, making feature extraction and matching more challenging. Since there is no established industry standard for constructing seamless reference databases over an area of interest (AOI). 

![decentrality_vis_perceptron](E:\MyPublications\manuscripts\3_SkyMAP-submitted-JPRS\代码仓库\figures\decentrality_vis_perceptron.jpg)

## 2. DReSS: Decentrality Related Street-view and Satellite-view dataset
DReSS dataset covers over 400 \(km^2\) in each of eight diverse cities around the world. The dataset consists of 422,760 aerial images sourced from Esri World Imagery, captured at zoom level 18 with a ground resolution of approximately 0.597 \(m\). Each aerial image has a resolution of (224 × 224) pixels. Additionally, DReSS features 174,934 street-view panoramas obtained using the Google Street View. These panoramas are randomly distributed within the coverage area of the aerial images, with an average interval of about 500 \(m\) between samples. The panoramas are North-aligned, and each has a resolution of (2048 × 1024) pixels. 

To download the DReSS dataset (including aerial images and IDs of street-view images), you can click: [🤗DReSS](https://huggingface.co/datasets/SummerpanKing/DReSS).

### City distribution

![dataset_locations](E:\MyPublications\manuscripts\3_SkyMAP-submitted-JPRS\代码仓库\figures\dataset_locations.jpg)



### Street-View Image download

To comply with Google Street View's policies, we only provide the **panorama IDs** along with instructions for downloading. This allows users to access the dataset independently.

With these IDs, users can retrieve the panoramas using the [Google Street View Static API](https://developers.google.com/maps/documentation/streetview?hl=zh-cn) or [third-party tools](https://svd360.com/).

**If you have any questions or encounter any issues during access or download, feel free to contact us** at [xiapanwang@whu.edu.cn].



## 3. Framework: AuxGeo



![framework](E:\MyPublications\manuscripts\3_SkyMAP-submitted-JPRS\代码仓库\figures\framework.jpg)



### Train the AuxGeo model
```python
# -- eg. VIGOR dataset

#-- 1. build the distance map.
python calc_distance_vigor.py

# -- 2. train the auxgeo model on vigor.
python train_vigor.py

# -- 3. evaluate the auxgeo model on vigor.
python eval_vigor_same.py
python eval_vigor_cross.py


# -- eg. DReSS dataset

#-- 1. build the distance map.
python calc_distance_skymap.py

# -- 2. train the auxgeo model on vigor.
python train_skymap.py

# -- 3. evaluate the auxgeo model on vigor.
python eval_skymap_same.py
python eval_skymap_cross.py

```



## Acknowledgments

This code is based on the amazing work of: [Sample4Geo](https://github.com/Skyy93/Sample4Geo) and [HC-Net](https://github.com/xlwangDev/HC-Net). We appreciate the previous open-source works.
## Citation✅
```
  @article{xia2024cross,
  title={Cross-View Geo-Localization with Street-View and VHR Satellite Imagery in Decentrality Settings},
  author={Xia, Panwang and Yu, Lei and Wan, Yi and Wu, Qiong and Chen, Peiqi and Zhong, Liheng and Yao, Yongxiang and Wei, Dong and Liu, Xinyi and Ru, Lixiang and others},
  journal={arXiv preprint arXiv:2412.11529},
  year={2024}
}
```

