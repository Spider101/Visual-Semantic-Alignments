import argparse
import requests
import shutil
from os import path, remove, listdir
from tqdm import tqdm

from config.resources import path_to_external_data


def make_visual_genome_dataset():
    region_desc_metadata_url = input("Enter URL for downloading region description metadata: ")
    image_metadata_url = input("Enter URL for downloading image metadata: ")

    path_to_save_data = path.join(path_to_external_data)
    path_to_save_region_desc_metadata = path.join(path_to_external_data, 'region_desc_metadata.json.zip')
    path_to_save_image_metadata = path.join(path_to_external_data, 'image_metadata.json.zip')

    print('\nStarting download of region descriptions metadata zip file...')
    region_metadata_response = requests.get(region_desc_metadata_url, stream=True)
    total_size_in_bytes= int(region_metadata_response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    # write the response to the region desc metadata zip file in local data folder
    with open(path_to_save_region_desc_metadata, 'wb') as fd:
        for chunk in region_metadata_response.iter_content(chunk_size=block_size):
            progress_bar.update(len(chunk))
            fd.write(chunk)
    progress_bar.close()

    # unpack and then remove the zip file
    print('\nUnzipping and then removing archive file for region description metadata...')
    shutil.unpack_archive(path_to_save_region_desc_metadata, path_to_save_data)
    remove(path_to_save_region_desc_metadata)

    print('Downloaded and unpacked data for Region Description Metadata')

    print('\nStarting download of image metadata zip file...')
    image_metadata_response = requests.get(image_metadata_url)
    total_size_in_bytes= int(image_metadata_response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    # write the response to the region desc metadata zip file in local data folder
    with open(path_to_save_image_metadata, 'wb') as fd:
        for chunk in image_metadata_response.iter_content(chunk_size=block_size):
            progress_bar.update(len(chunk))
            fd.write(chunk)
    progress_bar.close()

    # unpack and then remove the zip file
    print('\nUnzipping and then removing archive file for image metadata...')
    shutil.unpack_archive(path_to_save_image_metadata, path_to_save_data)
    remove(path_to_save_image_metadata)

    print('\nDownloaded and unpacked data for Image Metadata in the following locations:')
    for idx, file_path in enumerate(listdir(path_to_save_data)):
        print('{}. {}'.format(idx + 1, path.join(path_to_save_data, file_path)))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=None, help="which dataset to download")

    args = parser.parse_args()

    if args.dataset == "visual_genome":
        make_visual_genome_dataset()