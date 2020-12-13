# Overview

This repository is an exploration into the possibility of generating multi-sentence image descriptions by leveraging the latent dependencies between visual concepts in an image with their textual counterparts, optimized with a structured objective.

## Dataset

We will be combining information from the [Visual Genome dataset](http://visualgenome.org) and the dataset built as part of the [A Hierarchical Approach for Generating Descriptive Image Paragraphs](https://arxiv.org/pdf/1611.06607.pdf). To be more precise, we will be utilizing the bounding boxes and corresponding region descriptions from the former and the paragraph captions from the latter. This combined dataset will be a list of objects (19561) each having the following structure: `{"id" : int , "url": str, "paragraph": str , "regions": []}`. The *regions* key has a list of regions corresponding to that image in the following format: `{"image_id": "region_id": int, "x": int, "y": int, "height": int, "width": int, "phrase": str}`

## Prerequisites

To build the dataset for this project, there are some prerequisites that need to be satisfied:

1. This project has a dependency on python version 3.7+, so make sure the version matches the minor version at least. This is a good resource to install python-3.7 if you don't have that version - [How to Install Python 3.7 on Ubuntu 18.04](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/). Alternatively, if you are using a different version of python, try removing the `[requires]` sections from the Pipfile and then trying to install the dependencies using it (see below).

2. This project also uses _Pipenv_ for dependency management. So, run the following command to install it globally via `pip` because we need to use it as a command directly instead of via `python -m [package-name]`.
    ```shell
    sudo pip3 install pipenv
    ```

3. Once Pipenv is installed and the repository has been cloned, cd into the project directory and run the following commands to start a new `virtualenv` and install the relevant dependencies.
    ```shell
    pipenv shell # creates a new virtualenv if none already exists
    pipenv install --dev # installs all the dependencies from the Pipfile needed for development
    ```
## Getting Started

1. Clone the repository and create a data directory inside according to the tree shown below:

    ------------
        ├── data
            │   ├── external       <- Data from third party sources.
            │   ├── interim        <- Intermediate data that has been transformed.
            │   ├── processed      <- The final, canonical data sets for modeling.
            │   └── raw            <- The original, immutable data dump.
    ------------

2. Run the `make_dataset.py` script with the `--dataset` argument as shown below
    ```shell
    python3 make_dataset --dataset [dataset-name]
    ```
    You can use _visual\_genome_ for downloading the [Visual Genome](http://visualgenome.org/api/v0/api_home.html) dataset and _image\_paragraphs_ for the dataset used in the [A Hierarchical Approach for Generating Descriptive Image Paragraphs](http://visualgenome.org/static/data/dataset/paragraphs_v1.json.zip) paper.
    The script will prompt you for the download links for the various data sources. You can use the following: 
    * [region_description_metadata](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip)
    * [image_metadata](http://visualgenome.org/static/data/dataset/image_data.json.zip)
    * [paragraph_metadata](http://visualgenome.org/static/data/dataset/paragraphs_v1.json.zip)

3. Create `config` package (a directory with an `__init__.py` file) at the root of the project. Then create a `resources.py` inside which will house all our constants pointing to data sources and other information. To start with, you can add the following variables with whatever values seem right for your system -
  * `local_data_path`: representing the path to the data directory from the root of the project
  * `VG_REG_FNAME`: representing the file name of the unzipped __visual genome image metadata__ 
  * `VG_IMG_FNAME`: representing the file name of the unzipped __region description data__
  * `PARA_FNAME`: representing the file name of the unzipped __paragraph dataset__
  * `METADICT_FNAME`: representing the file name for the dataset created by combining the metadata from the `VG_IMG_FNAME` and `PARA_FNAME` files (See `organize_data.py` for more details) 

4. Run the `organize_data.py` file as follows: `python organize_data.py --op align_data`. You will find the final dataset under your *data* directory in the `combined_data.json` file

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
