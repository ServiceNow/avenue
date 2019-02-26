import avenue
import argparse
import os
from datetime import datetime
import scipy.misc
import json

"""
Avenue come with a Datasetloader for Pytorch (AvenueDataset), to be able to create those dataset from a on of the
Avenue gym environment, you can use this file.
"""

now = datetime.now()
timestamp = int(datetime.timestamp(now))


def convert_dict_np_to_list(d):
    new_dict = {}
    for key, value in d.items():
        new_dict[key] = value.tolist()
    return new_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="Circuit_v1", type=str,
                        help='environment to train on (default: "Circuit_v1")')

    parser.add_argument('--number_of_data', default=100, type=int,
                        help='Number of frames to collect (default: 100)')

    parser.add_argument('--save_path', default="/tmp", type=str,
                        help='Path where the dataset will be saved (default: "/tmp")')

    args = parser.parse_args()
    return args


args = get_args()
labels = []

full_directory_path = os.path.join(args.save_path, args.env_name + "_" + str(timestamp))

os.makedirs(full_directory_path, mode=0o777, exist_ok=False)

print("Your dataset will be saved in the directory: " + full_directory_path)

env = avenue.make(args.env_name)
env.reset()

for i in range(0, args.number_of_data):

    if i % 100 == 0:
        print (str(i) + " / " + str(args.number_of_data))

    ob, reward, done, info = env.step(env.action_space.sample())
    label_entry = convert_dict_np_to_list(info["avenue_state"]._asdict())
    image_path = "image_" + str(i) + ".jpg"
    label_entry["image"] = image_path
    scipy.misc.imsave(os.path.join(full_directory_path, image_path), ob["visual"])
    labels.append(label_entry)

    if done:
        env.reset()

    if i % 100 == 0:
        with open(os.path.join(full_directory_path, "labels.json"), 'w') as outfile:
            json.dump(labels, outfile)

print("Data successfully saved in : " + full_directory_path)