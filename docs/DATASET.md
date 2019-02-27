# Create a dataset

To create a new dataset composed of each image with the associated labels you can use our script (dataset_collection/
collect_avenue_dataset.py). This will execute the simulation and save the association of image with associated label
that you will be able to load in the future using pytorch Dataset module.

Here's the command to create a dataset, you can specify the dataset you want to generate data from, the size of 
the dataset and the directory where you want the dataset to be save:

```
python avenue/dataset_collection/collect_avenue_dataset.py --save_path /tmp --env_name ScenarioZoom_v1 --number_of_data 1000
```

# Load a dataset
Once you have created your dataset in a directory, you can now load youre dataset like this, all you have to do is
to specify the directory path of your dataset (and pytorch transform object eventually):

```
    from avenue.avenue_dataset import AvenueDataset
    avenue_data = AvenueDataset("/your/path/", transform = None)
```


For example, if you want the first item of your dataset, you can do:

```
first_data = avenue_data[0]
```

Or the length of your dataset:

```
len(avenue_data)
```

You can now manipulate the data with following documentation: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# What are the labels of each environment ?

To find the available labels of each environment you can refer to: 
[Documentation of available environments](ENVIRONMENTS.md)

# Available dataset

## OnRoadObjectClassification

Dataset that return images with object on the road from three classes (boxes, balls and trashes)
and the corresponding class of the object and distance.

### Download link
https://drive.google.com/file/d/1INRpo8r6NUDw8KcOyTsk3mwoe1COcHE0/view?usp=sharing

### How to load it ?
```
from avenue.avenue_dataset import OnRoadObjectClassification
dataset = OnRoadObjectClassification(root_dir=/dataset/path/, transform=None)
```

### What does it return ?
A tuple with as the first element the image of size (Width, Height, 1), object class (int between 0 and 2), 
object distance (in meters).


```
dataset[0]
# Return
(array([[[ 31],[ 39],[ 38],...,[ 68],[ 81],[ 79]]], dtype=uint8), 1, 138.14410400390625)
```