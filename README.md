# Overview

This repository is an exploration into the possibility of generating multi-sentence image descriptions by leveraging the latent dependencies between visual concepts in an image with their textual counterparts, optimized with a structured objective.

## Dataset

We will be combining information from the [Visual Genome dataset](http://visualgenome.org) and the dataset built as part of the [A Hierarchical Approach for Generating Descriptive Image Paragraphs](https://arxiv.org/pdf/1611.06607.pdf). To be more precise, we will be utilizing the bounding boxes and corresponding region descriptions from the former and the paragraph captions from the latter.

## Prerequisites

To build the dataset for this project, there are some prerequisites that need to be satisfied:

1. Download the metadata related to the dataset used in the [A Hierarchical Approach for Generating Descriptive Image Paragraphs](http://visualgenome.org/static/data/dataset/paragraphs_v1.json.zip) paper and unzip it to your *data* directory on your local drive.

2. Download the [image metadata](http://visualgenome.org/static/data/dataset/image_data.json.zip) and [region descriptions](http://visualgenome.org/static/data/dataset/image_data.json.zip) from the [Visual Genome](http://visualgenome.org) dataset and unzip it to your *data* directory on you local drive.

3. Setup a *config* directory on the same level as your *data* directory and create a `resources.py` file with the following variables - `data_path`, `VG_REG_FNAME`, `VG_IMG_FNAME` and `PARA_FNAME` that represent the *data directory*, filenames of the unzipped **visual genome image metadata**, **region description data** and **paragraph dataset** respectively (sans the json extension). 

4. Run the `organize_data.py` file as follows: `python organize_data.py --op align_data`

You will find the final dataset under your *data* directory in the `combined_data.json` file

## Bibliography

```
@inproceedings{krause2016paragraphs,
  title={A Hierarchical Approach for Generating Descriptive Image Paragraphs},
  author={Krause, Jonathan and Johnson, Justin and Krishna, Ranjay and Fei-Fei, Li},
  booktitle={Computer Vision and Patterm Recognition (CVPR)},
  year={2017}
}
```

```
@inproceedings{krishnavisualgenome,
  title={Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations},
  author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalanditis, Yannis and Li, Li-Jia and Shamma, David A and Bernstein, Michael and Fei-Fei, Li},
  year = {2016},
}
```
