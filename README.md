# brain_matrix

brain_matrix is a python package for distance metric oriented fMRI meta-analysis. The package provides one main class `BrainMatrix` which has fMRI images for rows and columns. The entries are distances between each images as defined by a given distance metric. By default, images are pulled from [Neurosynth](http://www.neurosynth.org) using keyword features like "semantic" or "anxiety". For a distance metric, we provide [Earth Movers Distance](
https://en.wikipedia.org/wiki/Earth_mover%27s_distance), a metric that has been successfully used to compare 2D images. However, one can also provide their own distance metric function.

## Usage
Using `BrainMatrix` is fairly straightforward. An example is the best explanation:

```python
from brain_matrix import BrainMatrix

if __name__ == '__main__':
    # We use an excessively large downsample value for demonstration
    matrix = BrainMatrix(downsample=30, name='example')
    matrix.compute_distances(['syntactic', 'speech', 'semantic', 'music'])
    print(matrix['semantic']['speech'].distance)
    # distance matrix in csv form
    matrix.write_csv()
    # create some figure in figs/
    matrix.plot_dendrogram()
    matrix.plot_mds(clusters=2, dim=3, interactive=True)
```

## How does this work and why should I care?
Most generally, this package allows the user to explore the similarity structure of cognitive activities in terms of their neural underpinnings. Given a set of cognitive activities, brain_matrix provides a scalar difference between any two activities. To facilitate understanding, these activities can then be embedded into a low dimensional space by multidimensional scaling. brain_matrix requires two independent pieces to provide this functionality:

1. A mapping from cognitive activities onto three dimensional arrays (i.e. three dimensional images).
2. A distance metric over three dimensional arrays such that distances in the image space reflect functional differences between the associated tasks.

fMRI images provide a clear option for (1). fMRI images are three dimensional arrays, where each element is a voxel, whose value reflects the activity level of the neurons in that location in the brain. If you have an fMRI scanner, you can create a mapping for any activity you can get a subject to do in the scanner. For the rest of us, brain_matrix provides an interface to Neurosynth, a meta-analysis platform. Neurosynth creates a meta-analytic fMRI image for a given keyword (e.g. "visual") by combining reported activations from papers in which the word ("visual") occurs frequently. Thus, we approximate a mapping from cognitive activities to brain activation with a mapping from words to activations reported in papers that contain those words.

For (2) brain_matrix provide Earth Movers Distance, as implemented by [Ofir Pele and Michael Werman ](http://www.ariel.ac.il/sites/ofirpele/fastemd/), using a python wrapper developed by [Will Mayner](https://github.com/wmayner/pyemd). EMD is a powerful metric for measuring the distances between multidimensional arrays because it respects the array's internal structure. If we flattened an fMRI image into a long one dimensional array, we could use Euclidean distance, however this would treat all voxels as completely independent, ignoring brain structure. To calculate the EMD between two images, we use a distance metric over the elements of the images themselves: a _voxel metric_. Activation in unique, but similar voxels contributes less to the overall image distance than activation in dissimilar voxels. We use "similar" and "dissimilar" here rather than "close" and "far" because the voxel metric need not reflect physical distance.

Recall from (2) that our image distance metric should be relevant to the cognitive tasks mapped onto each image. We will take as given the assumption underlying all fMRI research that the location and activity levels of neurons is functionally related to cognition. This assumption largely validates the use of EMD. However, we are left with the difficult task of choosing a voxel metric. _The degree to which EMD accurately reflects the functional differences of brain images is highly dependent on the degree to which our voxel metric reflects the functional differences of voxels._

The present implementation uses a simple and far from ideal voxel metric of Euclidean distance. That is, we assume that voxels are functionally similar to the degree to which they are spatially close. Superficially, this assumption is faulty because it ignores the physical structure of the brain, the gyri and sulci and whatnot. More seriously, however, this assumption ignores the connectivity structure of the brain. That is, two brain areas could be physically disparate, but functionally close if there is a fast path of communication between the two. _Creating a voxel metric based on connectivity rather than spatial distance would constitute a major improvement to this model._ Perhaps surprisingly, however, we find that this unsophisticated metric still gives highly intuitive results. This is likely because, in general, the brain attempts to minimize long range connections [citation neeeded].

Given (1) a mapping from keywords to brain images, and (2) a distance metric over brain images, the workings of brain_matrix are straightforward. The user provides a list of keywords that Neurosynth has indexed ad brain_matrix fetches an image for each one, computing the distances between each pair. This results in a distance matrix, which is prepared for human consumption in two forms: a two or three dimensional plot via multidimensional scaling, and a dendrogram via hierarchical clustering.

### Dendrogram
![dendrogram](http://imgur.com/6DGITZ7.png)

### Multidimensional scaling
![mds](http://imgur.com/zfG13O7.png)

## Process
A rough description of the processing pipeline:

- for each feature
    - get a list of studies that Neurosynth has labeled with this feature
    - get a composite fMRI image (using Neurosynth) for these studies
    - transform this image into a lower dimensional form using the `image_transform` function. This is a block reduction by default, but the user could provide an alternative (perhaps anatomically justified) transformation function.
- for each pair of features
    - get the image associated with each feature
    - compute the distance between the two features as defined by `metric`. By default, we use Earth Movers Distance. The user can provide her own function that two images in the form returned by `image_transform` (a three dimensional array by default).
