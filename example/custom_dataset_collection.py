import avenue
import argparse
from datetime import datetime
import scipy.misc
import json
from avenue.envs import *
from avenue.wrappers import *
import random

"""
Avenue come with a Datasetloader for Pytorch (AvenueDataset), to be able to create those dataset from a on of the
Avenue gym environment, you can use this file.
This is just an example, you might want to create your own datacollector with different variations.
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
    parser.add_argument('--env_name', default="Env_name", type=str,
                        help='environment to train on (default: "Circuit_v1")')

    parser.add_argument('--number_of_data', default=100, type=int,
                        help='Number of frames to collect (default: 100)')

    parser.add_argument('--save_path', default="/tmp", type=str,
                        help='Path where the dataset will be saved (default: "/tmp")')

    args = parser.parse_args()
    return args


args = get_args()
labels = []

# Create necessary directories to store the dataset
full_directory_path = os.path.join(args.save_path, args.env_name + "_" + str(timestamp))
directory_rgb = os.path.join(full_directory_path, "rgb")
directory_segmentation = os.path.join(full_directory_path, "segmentation")

os.makedirs(full_directory_path, mode=0o777, exist_ok=False)
os.makedirs(directory_rgb, mode=0o777, exist_ok=False)
os.makedirs(directory_segmentation, mode=0o777, exist_ok=False)


print("Your dataset will be saved in the directory: " + full_directory_path)

env = None


def CustomEnv(config=None, **kwargs):

    random_hour = random.randint(6, 20)
    if random_hour < 9 or random_hour > 17:
        night_mode = True
    else:
        night_mode = False

    # Randomize config here
    old_config = {
        "road_length": 500,
        "curvature": random.randint(0, 100),
        "lane_number": random.randint(1, 4),
        "task": 2,
        "time": random_hour,
        "city_seed": random.randint(0, 100000),
        "skip_frame": 30,
        "height": 600,
        "width": 800,
        "night_mode" : night_mode,
        "pedestrian_distracted_percent": 0,
        "pedestrian_density": 0,
        "weather_condition": 4
    }
    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)

    return env


# For the number of data
for i in range(0, args.number_of_data):

    # Recreate the environment every 100 step (useful because of the procedural generation, some environments
    # regenerate themselves at each "make").
    if i % 100 == 0:
        print (str(i) + " / " + str(args.number_of_data))
        if env is not None:
            print("Close")
            env.close()
            print("Closed")

        env = CustomEnv()

        env.reset(train_mode=True)

    ob, reward, done, info = env.step(env.action_space.sample())

    # Get the label data
    label_entry = convert_dict_np_to_list(info["avenue_state"]._asdict())
    image_path = "image_" + str(i)

    label_entry["image"] = image_path
    labels.append(label_entry)

    # Split rgb and segmentation
    segmentation = ob["visual"]["segmentation"]
    rgb = ob["visual"]["rgb"]

    # Save to the destination repository
    scipy.misc.imsave(os.path.join(directory_rgb, image_path) + ".jpg", rgb)
    scipy.misc.toimage(segmentation[:, :, 0], cmin=0.0, cmax=255, mode="I").save(os.path.join(directory_segmentation, image_path) + ".png")

    if done:
        env.reset()

    # Save the labels regulary
    if i % 100 == 0:
        with open(os.path.join(full_directory_path, "labels.json"), 'w') as outfile:
            json.dump(labels, outfile)

print("Data successfully saved in : " + full_directory_path)
