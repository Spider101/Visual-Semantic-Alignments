import argparse
import requests
import shutil
from os import path, remove, listdir
from tqdm import tqdm

from config.resources import path_to_external_data

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

def unpack_archive(archive_path, dataset_name, path_to_save_location):
    # unpack and then remove the zip file
    print('\nUnzipping and then removing archive file for {} metadata...'.format(dataset_name))
    shutil.unpack_archive(archive_path, path_to_save_location)
    remove(archive_path)

    print('Downloaded and unpacked data for Region Description Metadata')

def make_visual_genome_dataset():
    region_desc_metadata_url = input("Enter URL for downloading region description metadata: ")
    image_metadata_url = input("Enter URL for downloading image metadata: ")

    path_to_save_data = path.join(path_to_external_data)
    path_to_save_region_desc_metadata = path.join(path_to_external_data, 'region_desc_metadata.json.zip')
    path_to_save_image_metadata = path.join(path_to_external_data, 'image_metadata.json.zip')

    # build region descriptions metadata dataset
    region_desc_metadata_dataset_name = 'Region Description Metadata'
    download_from_url(region_desc_metadata_url, region_desc_metadata_dataset_name, path_to_save_region_desc_metadata)
    unpack_archive(path_to_save_region_desc_metadata, region_desc_metadata_dataset_name, path_to_save_data)

    # build image metadata dataset
    image_metadata_dataset_name = 'Image Metadata'
    download_from_url(image_metadata_url, image_metadata_dataset_name, path_to_save_image_metadata)
    unpack_archive(path_to_save_image_metadata, image_metadata_dataset_name, path_to_save_data)

    print('\nDownloaded and unpacked data for Region Description and Image Metadata in the following locations:')
    for idx, file_path in enumerate(listdir(path_to_save_data)):
        print('{}. {}'.format(idx + 1, path.join(path_to_save_data, file_path)))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=None, help="which dataset to download")

    args = parser.parse_args()

    if args.dataset == "visual_genome":
        make_visual_genome_dataset()