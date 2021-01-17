import argparse
import requests
import shutil
from os import path, remove, listdir, rename
from tqdm import tqdm

from config.resources import path_to_external_data, VG_IMG_FNAME, VG_REG_FNAME, PARA_FNAME

def download_from_url(url, dataset_name, path_to_save_location):
    print('\nStarting download of {} zip file...'.format(dataset_name))
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    # write the response to the region desc metadata zip file in local data folder
    with open(path_to_save_location, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=block_size):
            progress_bar.update(len(chunk))
            fd.write(chunk)
    progress_bar.close()

def unpack_archive(archive_path, dataset_name, path_to_save_dir, renamed_fname):
    # unpack and then remove the zip file
    print('\nUnzipping and then removing archive file for {} metadata...'.format(dataset_name))
    shutil.unpack_archive(archive_path, path_to_save_dir)
    remove(archive_path)

    # rename the extracted file to something project specific
    original_fname = path.basename(archive_path).split('.zip')[0]
    print('\nRenaming extracted file from {} to {}'.format(original_fname, renamed_fname))
    original_fpath = path.join(path_to_save_dir, original_fname)
    updated_fpath = path.join(path_to_save_dir, renamed_fname)
    rename(original_fpath, updated_fpath)

    print('Downloaded and unpacked data for Region Description Metadata')

def show_datasets_in_folder(path_to_folder):
    print('\nDownloaded and unpacked metadata files so far:')
    for idx, file_path in enumerate(listdir(path_to_folder)):
        print('{}. {}'.format(idx + 1, path.join(path_to_folder, file_path)))

def make_image_paragraphs_dataset():
    image_para_metadata_url = input("Enter URL for downloading image paragraphs metadata: ")

    path_to_save_image_para_metadata = path.join(path_to_external_data, 'paragraphs_v1.json.zip')

    # build image paragraphs metadata dataset
    image_paragraphs_metadata_name = 'Image Paragraphs Metadata'
    download_from_url(image_para_metadata_url, image_paragraphs_metadata_name, path_to_save_image_para_metadata)
    unpack_archive(path_to_save_image_para_metadata, image_paragraphs_metadata_name, path_to_external_data, PARA_FNAME)

    show_datasets_in_folder(path_to_external_data)

def make_visual_genome_dataset():
    region_desc_metadata_url = input("Enter URL for downloading region description metadata: ")
    image_metadata_url = input("Enter URL for downloading image metadata: ")

    path_to_save_region_desc_metadata = path.join(path_to_external_data, 'region_descriptions.json.zip')
    path_to_save_image_metadata = path.join(path_to_external_data, 'image_data.json.zip')

    # build region descriptions metadata dataset
    region_desc_metadata_name = 'Region Description Metadata'
    download_from_url(region_desc_metadata_url, region_desc_metadata_name, path_to_save_region_desc_metadata)
    unpack_archive(path_to_save_region_desc_metadata, region_desc_metadata_name, path_to_external_data, VG_REG_FNAME)

    # build image metadata dataset
    image_metadata_name = 'Image Metadata'
    download_from_url(image_metadata_url, image_metadata_name, path_to_save_image_metadata)
    unpack_archive(path_to_save_image_metadata, image_metadata_name, path_to_external_data, VG_IMG_FNAME)

    show_datasets_in_folder(path_to_external_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=None, help="which dataset to download")

    args = parser.parse_args()

    if args.dataset == "visual_genome":
        make_visual_genome_dataset()

    elif args.dataset == "image_paragraphs":
        make_image_paragraphs_dataset()