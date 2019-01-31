# Create a dataset

To create a new dataset composed of each image with the associated labels you can use our script (dataset_collection/
collect_avenue_dataset.py). This will execute the simulation and save the association of image with associated label
that you will be able to load in the future using pytorch Dataset module.

Here's the command to create a dataset, you can specify the dataset you want to generate data from, the size of 
the dataset and the directory where you want the dataset to be save:

```
python avenue/dataset_collection/collect_avenue_dataset.py --save_path /tmp --env_name ZoomScenario --number_of_data 1000
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