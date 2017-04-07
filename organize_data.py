import json
from os import listdir, mkdir
from os.path import join, exists
import argparse
import pdb

from config.resources import (data_path, VG_REG_FNAME, COCO_FNAME,
                                VG_IMG_FNAME, PARA_FNAME)

def align_datasets(d1, d2):

    counter, combined_data, num_para = 0, [], len(d1)
    print("Combining region and paragraph datasets. Please wait..\n")

    pdb.set_trace()
    for i in range(num_para):

        #collect paragraph data
        image_id = d1[i]["image_id"]
        url = d1[i]["url"]
        paragraph = d1[i]["paragraph"]
        
        entry = {"id": image_id, "url": url, "paragraph": paragraph}
        
        for j in range(len(d2)):

            if image_id == d2[j]["id"]:
                
                #add the regions metadata
                entry["regions"] = d2[j]["regions"]
                #add the combined entry to the list
                combined_data.append(entry)

                counter += 1
                print("{} out of {} urls matched".format(counter, num_para),
                                end="\r", flush=False,)

    print("\nCombining datasets completed!")
    return combined_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--op", default=None, help="operation to perform")

    args = parser.parse_args()

    if args.op == "align_data":
        
        para_dict = json.load(open(join(data_path, PARA_FNAME)))
        vg_img_dict = json.load(open(join(data_path, VG_REG_FNAME)))
        combined_data = align_datasets(para_dict, vg_img_dict)
        combined_data_path = join(data_path, "combined_data.json")
        json.dump(combined_data, open(combined_data_path, "w"))
