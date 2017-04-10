"""Tools for meta-analyses of feature-based fMRI activations

Main class is BrainMatrix which is a distance matrix of meta-analytic images
pulled from neurosynth. 
"""

from __future__ import division

from collections import defaultdict
import copy
import inspect
import logging
import multiprocessing as mp
import os
import sys
import time
from tqdm import tqdm
import joblib

import nibabel as nib
import numpy as np
import pandas as pd

from neurosynth.base.dataset import Dataset
from neurosynth.analysis.meta import MetaAnalysis

from distance import euclidean_emd, block_reduce
import plotting
from utils import lazy_property



# WARNING indicates beginning of computationally expensive processes
# INFO indicates results and run time of computations
# CRITICAL indicates major results
LOG = logging.getLogger()
LOG.setLevel(logging.ERROR)
LOG.addHandler(logging.NullHandler())  # allows turning off logging alltogether

printer = logging.StreamHandler()
printer.setFormatter(logging.Formatter(
                   datefmt='%X', fmt='[%(levelname)s]\t(%(asctime)s):\t%(message)s'))

filer = logging.FileHandler('log.txt')
filer.setFormatter(logging.Formatter(
                   datefmt='%X', fmt='[%(levelname)s]\t(%(asctime)s):\t%(message)s'))

LOG.addHandler(filer)
LOG.addHandler(printer)
del filer
del printer


class BrainMatrix(dict):
    """A distance matrix of fMRI meta-analysis images

    BrainMatrix is a dict from feature names to MetaImage objects.
    Distance between two features is given by self[feature1][feature2].
    Lazy evaluation is used throughout, so you can quickly set up the
    structure of your matrix before doing the heavy computational work.
    Compute distances for all Distance objects with the compute_distances()
    method.

    Attributes:
      metric: 'emd' or a function that takes two 3D arrays as argument
        and returns a float.
      image_type (str): the statistical method for creating intensity values
        in the image, e.g. pAgF is p(activation|feature)
      downsample (float/int): the factor by which images are downsampled before
        distances are measured.
      image_transform (str): method of downsampling.
        Can be 'block_reduce' or 'spline'
      blur (float/int): the sigma value for the gaussian filter applied to
        images before downsample when using block_reduce
      validation_trials (int): number of trials to use in cross validation
      auto_save (bool): should we save periodically during long computations?
      multi (bool): use multiprocessing?
      name (str): analysis is saved to cache/analyses/name.pkl, and can
        be loaded with load_brainmatrix(name)

    """
    def __init__(self, metric='emd', image_type='pAgF', name=None, multi=True,
                 image_transform='block_reduce', downsample=8, auto_save=True,
                 data=None):
        self.image_type = image_type
        self.multi = multi
        self.downsample = downsample
        self.auto_save = auto_save

        if callable(metric):
            self.metric = metric
        elif metric == 'emd':
            self.metric = euclidean_emd
        else:
            raise ValueError('{metric} is not a valid metric'.format(**locals()))

        if callable(image_transform):
            self.image_transform = image_transform
        elif image_transform == 'block_reduce':
            from functools import partial
            self.image_transform = partial(block_reduce, factor=downsample)
            #def block_reduce_transform(image):
                #"""The default transformation."""
                #return block_reduce(image, downsample, blur)
            #self.image_transform = block_reduce_transform
        else:
            raise ValueError(('{image_transform} is not a valid'
                              'transform function').format(**locals()))
        self.name = name if name else time.strftime('analysis_on_%m-%d_%H-%M-%S')

        if isinstance(data, Dataset):
            self.data = data
        elif isinstance(data, str):
            LOG.warning('Loading %s', data)
            self.data = Dataset.load(data)
        elif data is None:
            try:
                LOG.warning('Loading data/dataset.pkl')
                self.data = Dataset.load('data/dataset.pkl')
            except FileNotFoundError:
                self.data = _getdata()

    @property
    def features(self):
            return self.keys()

    def to_dataframe(self, features=None):
        """Returns distance matrix for `features` as a pandas dataframe.

        If features is None, use all features in self.
        """
        if not features:
            features = self.features

        data = defaultdict(dict)
        for f1 in features:
            for f2 in features:
                data[f1][f2] = self[f1][f2].distance

        df = pd.DataFrame.from_dict(data)
        return df

    def compute_distances(self, features, processes=None):
        """Computes distance between each feature in `features`.

        Only computes distances that have not already been computed. Utilizes
        multiprocessing to maximize speed. Processes will complete in batches
        with size equal to the number of cpu cores on the machine. Distances
        will be logged as they are computed.
        """        
        dists_to_compute = []
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                distance_value = self[f1][f2].distance
                if distance_value is None:
                    dists_to_compute.append(self[f1][f2])
        LOG.warning('Computing {} distances'.format(len(dists_to_compute)))
        img_pairs = [(d.image1.image, d.image2.image) for d in dists_to_compute]

        if not self.multi:
            # the simple way...
            for i, dist in enumerate(dists_to_compute):
                dist.distance = self.metric(*img_pairs[i])
        else:
            # equivalent to above, but with multiprocessing
            with mp.Pool(processes) as pool, tqdm(total=len(dists_to_compute)) as pbar:
                results = [pool.apply_async(self.metric, pair) for pair in img_pairs]
                for i, dist in enumerate(dists_to_compute):
                    dist.distance = results[i].get()  # blocks until the result is available
                    pbar.update(1)
                    if i % 8 == 7 and self.auto_save:
                        self.save()

        LOG.warning('All {} distances computed.'.format(len(dists_to_compute)))
        if self.auto_save:
            self.save()

    def plot_mds(self, features=None,
     dim=2, metric=True,
                 clustering=True, clusters=4, interactive=False):
        """Saves a scatterplot of the features projected onto 2 dimensions.

        Uses MDS to project features onto a 2 or 3 dimensional based on their
        distances from each other. If features is not None, only plot those
        that are given. If interactive is truthy, an interactive plot will pop up. This is
        recommended for 3D graphs which are hard to make sense of without
        rotating the graph.
        """
        df = self.to_dataframe(features)
        stress = plotting.mds(df, name=self.name, dim=dim, metric=metric,
                              clustering=clustering, clusters=clusters,
                              interactive=interactive)
        return stress

    def plot_dendrogram(self, features=None, method='complete'):
        """Plots a dendrogram using hierarchical clustering.

        see scipy.cluster.hierarchy.linkage for details regarding
        possible clustering methods.
        """
        df = self.to_dataframe(features)
        inconsistency = plotting.dendrogram(df, name=self.name, method=method)
        return inconsistency

    def write_csv(self, features=None):
        """Creates distances.csv, a distance matrix of all MetaImages in self."""
        df = self.to_dataframe(features)
        df.to_csv('distances.csv')

    def save(self):
        """Saves self to cache/analyses/{self.name}.pkl for future retrieval."""
        LOG.debug('Attempting to save brain matrix')
        # delete extraneous information to keep files small
        save = copy.copy(self)
        del save.data
        save = copy.deepcopy(save)  # so we don't change features
        for feature in save.values():
            try:
                del feature._lazy_image
            except AttributeError:
                pass

        os.makedirs('cache/analyses', exist_ok=True)
        file = ('cache/analyses/{}.pkl').format(self.name)
        joblib.dump(save, file, compress=3)
        LOG.info('BrainMatrix saved to {}'.format(file))


    def __missing__(self, key):
        # constructor will raise an error if key is not a valid feature
        return MetaImage(key, self)

    def __str__(self):
        return 'BrainMatrix(features={})'.format(list(self.features))
        # return ('BrainMatrix(image_type={image_type}, metric={metric},\n'
                # '            image_transform={image_transform}, downsample={downsample},\n'
                # '            name={name})').format(**self.__dict__)

    def __repr__(self):
        return str(self)


class MetaImage(dict):
    """An fMRI metanalysis image with distances to other MetaImage objects.

    Attributes:
        feature (str): The keyword which is used to find relevant studies in
          the neurosynth database. Also a unique identifier of the MetaImage.
        bm (BrainMatrix): The parent BrainMatrix of this MetaImage
        img_file (str): The location of a .nii file. If none is given the file
          will be automatically downloaded from neurosynth.
    """
    def __init__(self, feature, bm, img_file=None):
        LOG.debug('Calling MetaImage({})'.format(feature))
        if ' ' in feature:
            raise ValueError('No spaces allowed in feature names.')
        self.feature = feature
        self.bm = bm
        bm[feature] = self  # add pointer from bm to this feature
        self[feature] = Distance(self.bm, self, self)  # dist to self
        if img_file:
            if not os.path.isfile(self.file):
                raise IOError("No such file: '{}'".format(img_file))
            self.img_file = img_file
        else:
            ns_name = feature.replace('_', ' ')   # neurosynth uses spaces in features
            if ns_name not in bm.data.get_feature_names():
                raise ValueError('No feature "{}" found in dataset'.format(ns_name))
            # use neurosynth file, will download later if needed
            self.img_file = 'data/{feature}_{bm.image_type}.nii.gz'.format(**self.__dict__)

    @lazy_property
    def studies(self):
        """Returns a list of IDs for studies in which the keyword occurs frequently.

        This will raise an exception if the image wasn't pulled from Neurosynthself."""
        ns_name = self.feature.replace('_', ' ')
        return self.bm.data.get_studies(features=ns_name)

    @lazy_property
    def image(self):
        """Returns a 3d brain image, the composition of activations in self.studies

        Exactly how the image is computed depends on self.image_type. See
        Neurosynth documentation for details. In short, the values in the
        image are probabilities associated with activation of each voxel
        and the feature being tagged to a study in self.studies

        The image is preprocessed using `self.bm.image_transform`, a function
        that maps the full fMRI image into a lower dimensional form. This reduction
        is necessary for Earth Mover's Distance to be tractable.
        """
        if not os.path.isfile(self.img_file):
            # get the image from neurosynth
            ma = MetaAnalysis(self.bm.data, self.studies)  # Neurosynth is so easy!
            ma.save_results('data/', self.feature, image_list=[self.bm.image_type])
        
        # load the image
        image = nib.load(self.img_file)
        image = np.array(image.dataobj)  # nibal uses array proxies

        return self.bm.image_transform(image)
        
    #@lazy_property
    #def cross_validation(self):
    #    """Returns cross validation for feature.

    #    This is meant to be a measure of the variance of the image
    #    associated with self. Cross validation is done by repeatedly
    #    splitting self.studies in half and computing the distance
    #    between the two images generated from the studies. The mean
    #    and variance of these distances are returned.
    #    """
    #    raise NotImplementedError('TOOD')

    #    LOG.warning('Calling {}.cross_validation()'.format(self))
    #    start = time.time()
    #    image_pairs = []
    #    studies = self.studies[:]  # copy the list
    #    for n in range(self.bm.validation_trials):
    #        random.shuffle(studies)

    #        studies1 = studies[:len(studies) // 2]
    #        studies2 = studies[len(studies) // 2:]
    #        ma1 = MetaAnalysis(self.bm.data, ids=studies1)
    #        ma2 = MetaAnalysis(self.bm.data, ids=studies2)

    #        # remove the mask to get a full cube image
    #        image1 = self.bm.data.masker.unmask(ma1.images[self.bm.image_type])
    #        image2 = self.bm.data.masker.unmask(ma2.images[self.bm.image_type])

    #        image_pairs.append((image1, image2))

    #    if self.multi:
    #        pool = mp.Pool()
    #        out = [pool.apply_async(earth_movers_distance, pair) for pair in image_pairs]
    #        for p in out:
    #            while not p.ready():
    #                # check periodically for keyboard interrupt
    #                try:
    #                    time.sleep(10)
    #                except KeyboardInterrupt:
    #                    # ask for user input
    #                    pool.terminate()
    #                    pool.join()
    #                    raise KeyboardInterrupt()

    #        distances = [p.get() for p in out]
    #        pool.close()
    #    else:
    #        distances = [earth_movers_distance(pair[0], pair[1], self.bm.downsample, 
    #                          self.bm.image_transform, self.bm.blur)
    #                     for pair in image_pairs]

    #    result = tuple(stats.describe(distances)[2:4])
    #    LOG.info('Returned {} in {:.0f} seconds'.format(result, time.time() - start))
    #    return result

    def __missing__(self, key):
        return Distance(self.bm, self, self.bm[key])

    def __repr__(self):
        return 'MetaImage({})'.format(self.feature)

    def __str__(self):
        return self.__repr__()


class Distance(dict):
    """Distance between two MetaImages. Keys indicate different metrics.

    Attributes:
      bm (BrainMatrix): the parent BrainMatrix for self
      image1 (MetaImage): distances are between this object and self.image2
      image2 (MetaImage): distances are between this object and self.image1
      distance (float): the value for the distance
      """
    def __init__(self, bm, image1, image2):
        super(Distance, self).__init__()
        self.bm = bm
        self.image1 = image1
        self.image2 = image2
        image1[image2.feature] = self  # distances are symmetric
        image2[image1.feature] = self
        if image1 is image2:
            self.distance = 0.0
        else:
            self.distance = None  # computed later

    @property
    def studies_jaccard_distance(self):
        s1 = set(self.image1.studies)
        s2 = set(self.image2.studies)
        return len(s1 & s2) / len(s1 | s2) 

    #def cross_validation(self, metric):
    #    """Returns cross validation for self

    #    Cross validation method is similar to that of MetaImage. Two images
    #    are repeatedly generated from halves of each feature image. The
    #    distance is computed between each image. The mean and variance
    #    of these distances are returned."""
    #    raise NotImplementedError('TOOD')
    #    LOG.warning("Calling ['{}']['{}'].cross_validation()".format(
    #                 self.image1, self.image2))
    #    start = time.time()
    #    image_pairs = []
    #    for _ in range(self.bm.validation_trials):
    #        studies1 = self.image1.studies[:]  # copy the list
    #        studies2 = self.image2.studies[:]  # copy the list
    #        random.shuffle(studies1)
    #        random.shuffle(studies2)
    #        studies1 = studies1[:len(studies1) // 2]
    #        studies2 = studies2[:len(studies2) // 2]
    #        ma1 = MetaAnalysis(self.bm.data, ids=studies1)
    #        ma2 = MetaAnalysis(self.bm.data, ids=studies2)
    #        # remove the mask to get a full cube image
    #        image1 = self.bm.data.masker.unmask(ma1.images[self.bm.image_type])
    #        image2 = self.bm.data.masker.unmask(ma2.images[self.bm.image_type])
    #        image_pairs.append((image1, image2))

    #    if MULTI:
    #        pool = mp.Pool()
    #        assert False  # metric
    #        out = [pool.apply_async(metric, args=(pair[0], pair[1], self.bm.downsample,
    #                                            self.bm.image_transform, self.bm.blur))
    #               for pair in image_pairs]
    #        for p in out:
    #            while not p.ready():
    #                # check periodically for keyboard interrupt
    #                try:
    #                    time.sleep(10)
    #                except KeyboardInterrupt:
    #                    pool.terminate()
    #                    pool.join()
    #                    raise KeyboardInterrupt()

    #        distances = [p.get() for p in out]
    #        pool.close()
    #    else:
    #        distances = [earth_movers_distance(pair[0], pair[1], self.bm.downsample,
    #                          self.bm.image_transform, self.bm.blur)
    #                     for pair in image_pairs]

    #    result = tuple(stats.describe(distances)[2:4])
    #    LOG.info('Returned {} in {} seconds'.format(result, time.time() - start))
    #    return result

    def __repr__(self):
        f1, f2 = self.image1.feature, self.image2.feature
        return 'Distance({f1}, {f2})'.format(**locals())

    def __str__(self):
        dct = super(Distance, self).__str__()
        rpr = self.__repr__()
        return '{rpr} =\n{dct}'.format(**locals())
        return self.__repr__


def load_brainmatrix(name):
    """Loads a cached BrainMatrix with same attribute values.

    name must refer to a .pkl file in cache/analysis, the
    location that BrainMatrix.save() deposits to."""
    LOG.debug('Attempting to load brain matrix')

    if name.endswith('.pkl'):
        name = name[:-4]

    file = 'cache/analyses/{}.pkl'.format(name)
    old = joblib.load(file)
    LOG.info('BrainMatrix loaded from file: {}'.format(file))
    return old


def _getdata():
    """Downloads data from neurosynth and returns it as a Dataset.

    Also pickles the dataset for future use."""
    LOG.warning('Downloading and processing Neurosynth database.')
    
    os.makedirs('data', exist_ok=True)
    from neurosynth.base.dataset import download
    download(path='data', unpack=True)
    
    data = Dataset('data/database.txt')
    data.add_features('data/features.txt')
    data.save('data/dataset.pkl')
    return data
