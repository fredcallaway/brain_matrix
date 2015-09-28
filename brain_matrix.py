"""Tools for meta-analyses of feature-based fMRI activations

Main class is BrainMatrix which is a distance matrix of meta-analytic images
pulled from neurosynth. 
"""

from __future__ import division

import copy
import inspect
import logging
import multiprocessing as mp
import os
import random
import shutil
import signal
import sys
import time

import IPython
import joblib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from nipy import load_image
import numpy as np
import pandas as pd
from scipy import stats, ndimage, spatial, cluster
from skimage.measure import block_reduce
from sklearn import cluster as skcluster, cross_decomposition, manifold

from neurosynth import Dataset, meta
import pyemd

MULTI = True
SAVE = True
labels_and_points = []  # a hack to get around namespace problems

# WARNING indicates beginning of computationally expensive processes
# INFO indicates results and run time of computations
# CRITICAL indicates major results
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
LOG.addHandler(logging.NullHandler())  # allows turning off logging alltogether

printer = logging.StreamHandler()
printer.setFormatter(logging.Formatter(
                   datefmt='%X', fmt='[%(levelname)s]\t(%(asctime)s)`\t%(message)s'))

filer = logging.FileHandler('log.txt')
filer.setFormatter(logging.Formatter(
                   datefmt='%X', fmt='[%(levelname)s]\t(%(asctime)s):\t%(message)s'))

LOG.addHandler(filer)
# LOG.addHandler(printer)
del filer
del printer



class BrainMatrix(dict):
    """A distance matrix of fMRI meta-analysis images

    BrainMatrix is a dict from feature names to Feature objects.
    Distance between two features is given by self[feature1][feature2].
    Lazy evaluation is used throughout, so you can quickly set up the
    structure of your matrix before doing the heavy computational work.
    Compute distances for all Distance objects with the compute_distances()
    method.

    Attributes:
      image_type (str): the statistical method for creating intensity values
        in the image.
      reduction (float/int): the factor by which images are downsampled before
        distances are measured.
      reduction_method (str): method of downsampling.
        Can be 'block_reduce' or 'spline'
      blur (float/int): the sigma value for the gaussian filter applied to
        images before reduction when using block_reduce
      validation_trials (int): number of trials to use in cross validation
    """
    def __init__(self, image_type='pAgF', reduction=5,
                 reduction_method='block_reduce', blur=None, validation_trials=16):
        super(BrainMatrix, self).__init__()
        self.image_type = image_type
        self.reduction = reduction
        self.reduction_method = reduction_method
        self.validation_trials = validation_trials

        if blur:
            self.blur = blur
        else:
            self.blur = round(2 * reduction / 6)

        self.data = Dataset.load('data/dataset.pkl')
        me = self.__dict__
        self.file = ('cache/analyses/{image_type}_{reduction}_'.format(**me) +
            '{reduction_method}_{blur}_{validation_trials}.bm'.format(**me))
        self._load()

    def get_df(self, features=None, distance_measure='emd'):
        """Returns distance matrix for `fetures` as a pandas dataframe.

        If features is None, use all features in self.
        """
        d = {}
        for f in self.values():
            row = {}
            for feature, distance in f.items():
                if distance_measure == 'emd':
                    try:
                        row[feature] = distance.distance
                    except AttributeError:
                        row[feature] = distance
                elif distance_measure == 'peak':
                    row[feature] = distance.peak_distance
                else:
                    raise ValueError("distance_measure must be 'emd' or 'peak'")
            d[f.name] = row
        df = pd.DataFrame.from_dict(d)

        if features is not None:
            df = df[features].ix[features]

        return df

    def compute_distances(self):
        """Computes all Distances in self.

        Only computes distances that have not already been computed. Utilizes
        multiprocessing to maximize speed. Processes will complete in batches
        with size equal to the number of cpu cores on the machine. Distances
        will be logged as they are computed.
        """
        dists_to_compute = []
        for feature in self.values():
            for distance in feature.values():
                if (distance not in dists_to_compute 
                    and (distance.distance is None)):
                        dists_to_compute.append(distance)
        img_pairs = [(d.feature1.image, d.feature2.image) for d in dists_to_compute]

        LOG.warning('Computing {} distances'.format(len(dists_to_compute)))
        pool = mp.Pool(initializer=_init_worker)
        results = [pool.apply_async(_emd, args=(img1, img2, self.reduction,
                                                self.reduction_method, self.blur))
                   for (img1, img2) in img_pairs]
        for i, d in enumerate(dists_to_compute):
            # get the results as they are returned
            start = time.time()

            while not results[i].ready():
                # periodically check for a keyboard interrupt
                try:
                    time.sleep(10)
                except KeyboardInterrupt:
                    IPython.embed()  # interactive shell
                    terminate = raw_input('Terminate computation? (yes|[no]): ')
                    if terminate == 'yes':
                        pool.terminate()
                        pool.join()
                        raise KeyboardInterrupt()

            d.distance = results[i].get()  # save value in the Distance object
            LOG.info('{}-{}: {}\t({} seconds)'.format(d.feature1, d.feature2,
                                                      d.distance, time.time() - start))
            if i % 8 == 7:
                LOG.info('{} out of {} distances computed'.format(i, len(dists_to_compute)))
                if SAVE:
                    # save data periodically to prevent catastrophic loss
                    # in the event of a crash
                    self.save()
        self.save()
        pool.close()

    def plot(self, features=None, dim=2, clustering='agglomerative', clusters = 4,
             distance_measure='emd'):
        """Displays a scatterplot of the features projected onto 2 dimensions.

        Uses MDS to project features based on their distances from each other.
        If features is not None, only plot those that are given.
        """
        df = self.get_df(features, distance_measure)[features].ix[features]
        if features is None:
            features = df.index

        if clustering is 'agglomerative':
            clustering = skcluster.AgglomerativeClustering(linkage='complete', 
                                                 affinity='precomputed', n_clusters=clusters)
            assignments = clustering.fit_predict(df)
        
        if dim == 2:
            # MDS to get points
            mds = manifold.MDS(n_components=2, eps=1e-9, dissimilarity="precomputed")
            points = mds.fit(df).embedding_

            plt.scatter(points[:,0], points[:,1], c=assignments, s=40)
            for label, x, y in zip(features, points[:, 0], points[:, 1]):
                plt.annotate(label, xy = (x, y), xytext = (-5, 5),
                             textcoords = 'offset points', ha = 'right', va = 'bottom')
        elif dim == 3:
            mds = manifold.MDS(n_components=3, eps=1e-9, dissimilarity="precomputed")
            points = mds.fit(df).embedding_

            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            xs, ys, zs = np.split(points, 3, axis=1)

            ax.scatter(xs,ys,zs, c=assignments, s=40)

            global labels_and_points  # a hack for namespace problems
            labels_and_points = []
            for feature, x, y, z in zip(features, xs, ys, zs):
                x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
                label = plt.annotate(
                    feature, 
                    xy = (x2, y2), xytext = (-5, 5),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',)
                labels_and_points.append((label, x, y, z))

            def update_position(e):
                for label, x, y, z in labels_and_points:
                    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
                    label.xy = x2,y2
                    label.update_positions(fig.canvas.renderer)
                fig.canvas.draw()

            fig.canvas.mpl_connect('motion_notify_event', update_position)

        plt.show()

    def plot_dendrogram(self, features=None, method='complete', distance_measure='emd'):
        """Plots a dendrogram using hierarchical clustering."""
        df = self.get_df(features, distance_measure)
        clustering = cluster.hierarchy.linkage(df, method=method)
        cluster.hierarchy.dendrogram(clustering, orientation='left', truncate_mode=None,
                                     labels=features, color_threshold=0)
        LOG.critical('INCONSISTENCY: {}'
                     .format(cluster.hierarchy.inconsistent(clustering)))
        plt.tight_layout()
        try:
            plt.savefig('figs/{}-dendrogram.png'.format(distance_measure))
        except IOError:
            os.mkdir('figs')
            plt.savefig('figs/{}-dendrogram.png'.format(distance_measure))


    def canonical_correlation_analysis(self, features):
        studies = sum([self[f].studies for f in features], [])  # concatenates paper lists
        studies = list(set(studies))  # remove duplicates

        feature_table = self.data.feature_table.data  # studies X features
        feature_table = feature_table[features]  # keep only our features
        feature_table = feature_table.ix[studies]  # keep only our studies
        # combine groups of studies into studygroups?


        def get_images(studies):
            masked_images = self.data.get_image_data(ids=studies).T  # studies X voxels
            # neurosynth removes non-brain voxels from flattened images
            # so we have to put them back to do block reduction
            images = [self.data.masker.unmask(img) for img in masked_images]  # a list of 3d images
            images = [_block_reduce(img, self.reduction, self.blur) for img in images]
            images = [img.ravel() for img in images]  # reflatten images
            images = np.array(images)

            return images

        images = get_images(studies)

        def categorize(feature_table):
            return [0 if row[0] > 0 else 1 for row in feature_table.as_matrix()]

        categories = categorize(feature_table)
        
        import IPython; IPython.embed()
        from neurosynth.analysis.classify import classify

        print classify(images, categories)
        print classify(images, categories, clf_method='SVM')
        print classify(images, categories, clf_method='Dummy')


        cca = cross_decomposition.CCA()
        print cca.fit(images, feature_table).score(images, feature_table)
        binary_table = np.ceil(feature_table)
        print cca.fit(images, binary_table).score(images, binary_table)

        import IPython; IPython.embed()

    def write_csv(self, features=None):
        """Creates distances.csv, a distance matrix of all Features in self"""
        df = self.get_df()
        if features is 'full':
            raise NotImplementedError('not done yet')
            full_features = df.dropna().index  # features with all distances computed
            df = df[full_features].ix[full_features]  # index by cols and rows
        elif features:
            df = df[features].ix[features]  # index by cols and rows
        df.to_csv('distances.csv')

    def save(self):
        """Saves self to file for future retrieval"""
        LOG.debug('Calling save({})'.format(locals()))
        # delete extraneous information to keep files small
        save = copy.copy(self)
        del save.data
        save = copy.deepcopy(save)  # so we don't change features
        for feature in save.values():
            try:
                del feature._lazy_image  # affects self
            except AttributeError:
                pass
        joblib.dump(save, self.file, compress=3)
        LOG.info('BrainMatrix saved to {}'.format(self.file))

    def _load(self):
        """Loads a cached BrainMatrix with same attribute values"""
        LOG.debug('Calling _load({})'.format(locals()))
        try:
            old = joblib.load(self.file)
        except IOError:
            # no cached BrainMatrix
            pass
        else:
            shutil.copyfile(self.file, self.file+'BACKUP')
            self.update(old)  # all distances in old are added to self

    def __missing__(self, key):
        # lazy evaluation
        return Feature(key, self)

    def __str__(self):
        me = self.__dict__
        return ("BrainMatrix(image_type='{image_type}', reduction=".format(**me) +
                "{reduction}, reduction_method='{reduction_method}', ".format(**me) +
                "blur={blur} validation_trials={validation_trials})".format(**me))


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated"""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


class Feature(dict):
    """An fMRI metanalysis image with distances to other Feature objects

    Attributes:
        name (str): The keyword which is used to find relevant studies in
          the neurosynth database.
        image_type (str): The type of analysis image to be used. As in neurosynth,
          this defaults to pFgA_z, i.e. p(feature|activation). This is
          operationalized as the probability that `name` occurs with
          frequency > 0.001 given that activation of a voxel is present.
        studies (list): Studies in which self.name appears more than .1%
          of the time. Reported activations are used to generate images.
    """
    def __init__(self, name, bm):
        super(Feature, self).__init__()
        LOG.debug('Calling Feature({})'.format(locals()))
        if ' ' in name:
            raise ValueError('No spaces allowed in feature names.')
        self.name = name
        self.bm = bm
        bm[name] = self  # add pointer from bm to this feature
        ns_name = name.replace('_', ' ')   # neurosynth uses spaces in features
        if ns_name not in bm.data.get_feature_names():
            raise ValueError('No feature "%s" found in dataset' % ns_name)
        self[name] = Distance(self.bm, self, self)  # dist to self

    @lazy_property
    def studies(self):
        """Returns a list of study ID numbers"""
        LOG.debug('Calling studies({})'.format(locals()))
        ns_name = self.name.replace('_', ' ')
        return self.bm.data.get_ids_by_features(ns_name)

    @lazy_property
    def cross_validation(self):
        """Returns cross validation for feature.

        This is meant to be a measure of the variance of the image
        associated with self. Cross validation is done by repeatedly
        splitting self.studies in half and computing the distance
        between the two images generated from the studies. The mean
        and variance of these distances are returned.
        """
        LOG.warning('Calling {}.cross_validation()'.format(self))
        start = time.time()
        image_pairs = []
        for n in range(self.bm.validation_trials):
            studies = self.studies[:]  # copy the list
            random.shuffle(studies)

            studies1 = studies[:len(studies) // 2]
            studies2 = studies[len(studies) // 2:]
            ma1 = meta.MetaAnalysis(self.bm.data, ids=studies1)
            ma2 = meta.MetaAnalysis(self.bm.data, ids=studies2)

            # remove the mask to get a full cube image
            image1 = self.bm.data.masker.unmask(ma1.images[self.bm.image_type])
            image2 = self.bm.data.masker.unmask(ma2.images[self.bm.image_type])

            image_pairs.append((image1, image2))

        if MULTI:
            pool = mp.Pool(initializer=_init_worker)
            out = [pool.apply_async(_emd, args=(pair[0], pair[1], self.bm.reduction,
                                          self.bm.reduction_method, self.bm.blur))
                   for pair in image_pairs]
            for p in out:
                while not p.ready():
                    # check periodically for keyboard interrupt
                    try:
                        time.sleep(10)
                    except KeyboardInterrupt:
                        # ask for user input
                        pool.terminate()
                        pool.join()
                        raise KeyboardInterrupt()

            distances = [p.get() for p in out]
            pool.close()
        else:
            distances = [_emd(pair[0], pair[1], self.bm.reduction, 
                              self.bm.reduction_method, self.bm.blur)
                         for pair in image_pairs]

        result = tuple(stats.describe(distances)[2:4])
        LOG.info('Returned {} in {} seconds'.format(round(result, time.time() - start)))
        return result

    @lazy_property
    def image(self):
        """Returns a 3d brain image, the composition of activations in self.studies

        Exactly how the image is computed depends on self.image_type. See
        Neurosynth documentation for details. In short, the values in the
        image are probabilities associated with activation of each voxel
        and the feature being tagged to a study in self.studies
        """
        LOG.debug('Calling image({})'.format(locals()))
        if not os.path.isfile('data/%s_%s.nii' % (self.name, self.bm.image_type)):
            ma = meta.MetaAnalysis(self.bm.data, self.studies)
            # save to files
            ma.save_results('data', self.name, image_list=[self.bm.image_type])
            os.system('gunzip data/%s_%s.nii' % (self.name, self.bm.image_type))
        # load the image
        image = load_image('data/%s_%s.nii' % (self.name, self.bm.image_type))

        return image
        
    def __missing__(self, key):
        return Distance(self.bm, self, self.bm[key])

    def __str__(self):
        return str(self.name)


class SubFeature(Feature):
    """Represents a subset of studies tagged with a feature."""
    def __init__(self, name, bm, studies):
        super(SubFeature, self).__init__(name, bm)
        bm[name] = self  # add pointer from bm to this feature
        ns_name = name.replace('_', ' ')   # neurosynth uses spaces in features
        if ns_name not in bm.data.get_feature_names():
            raise ValueError('No feature "%s" found in dataset' % ns_name)
        self[name] = Distance(self.bm, self, self)  # dist to self
        self.studies = studies
        

class Distance(object):
    """Earth Mover's Distance between two feature images

    Attributes:
      bm (BrainMatrix): the parent BrainMatrix for self
      feature1 (Feature): one feature that self represents the distance between
      feature2 (Feature): the other feature that self represents the distance between
      distance: the Earth Mover's Distance between feature1 and feature2
      """
    def __init__(self, bm, feature1, feature2):
        super(Distance, self).__init__()
        LOG.debug('Calling Disance.__init__({})'.format(locals()))
        self.bm = bm
        self.feature1 = feature1
        self.feature2 = feature2
        feature1[feature2.name] = self  # add pointers from features to self
        feature2[feature1.name] = self
        if feature1 is feature2:
            self.distance = 0  # dist to self is 0
        else:
            # computation of distances is deferred until compute_distances is called
            self.distance = None

    @lazy_property
    def peak_distance(self):
        """Returns Euclidean distance between the peaks of each image."""

        def peak(img):
            """Returns coordinates of voxel with highest activation in img"""
            flat_img = np.array(img).flatten()
            inds = np.argpartition(flat_img, -100)[-100:]
            inds = [ind for ind in inds if flat_img[ind] == np.max(flat_img[inds])]
            coords = np.array(np.unravel_index(inds, img.shape)).T
            assert len(coords) == 1  # there is a unique peak
            return coords[0]

        # we reduce the images so that the peak and edm distances have same input
        images = [_block_reduce(img, self.bm.reduction, self.bm.blur)
                  for img in (self.feature1.image, self.feature2.image)]

        peaks = np.array([peak(img) for img in images])
        return spatial.distance.pdist(peaks)[0]


    @lazy_property
    def cross_validation(self):
        """Returns cross validation for self

        Cross validation method is similar that of Feature. Two images
        are repeatedly generated from halves of each feature image. The
        distance is computed between each image. The mean and variance
        of these distances are returned."""
        LOG.warning("Calling ['{}']['{}'].cross_validation()".format(
                     self.feature1, self.feature2))
        start = time.time()
        image_pairs = []
        for _ in range(self.bm.validation_trials):
            studies1 = self.feature1.studies[:]  # copy the list
            studies2 = self.feature2.studies[:]  # copy the list
            random.shuffle(studies1)
            random.shuffle(studies2)
            studies1 = studies1[:len(studies1) // 2]
            studies2 = studies2[:len(studies2) // 2]
            ma1 = meta.MetaAnalysis(self.bm.data, ids=studies1)
            ma2 = meta.MetaAnalysis(self.bm.data, ids=studies2)
            # remove the mask to get a full cube image
            image1 = self.bm.data.masker.unmask(ma1.images[self.bm.image_type])
            image2 = self.bm.data.masker.unmask(ma2.images[self.bm.image_type])
            image_pairs.append((image1, image2))

        if MULTI:
            pool = mp.Pool(initializer=_init_worker)
            out = [pool.apply_async(_emd, args=(pair[0], pair[1], self.bm.reduction,
                                          self.bm.reduction_method, self.bm.blur))
                   for pair in image_pairs]
            for p in out:
                while not p.ready():
                    # check period#5872A2ically for keyboard interrupt
                    try:
                        time.sleep(10)
                    except KeyboardInterrupt:
                        # ask for user input
                        pool.terminate()
                        pool.join()
                        raise KeyboardInterrupt()

            distances = [p.get() for p in out]
            pool.close()
        else:
            distances = [_emd(pair[0], pair[1], self.bm.reduction,
                              self.bm.reduction_method, self.bm.blur)
                         for pair in image_pairs]

        result = tuple(stats.describe(distances)[2:4])
        LOG.info('Returned {} in {} seconds'.format(result, time.time() - start))
        return result

    def __str__(self):
        return str(self.distance)


class PaperMatrix(object):
    """A clustering of papers in severel studies"""
    def __init__(self, features, num_papers):
        super(PaperMatrix, self).__init__()
        self.features = features
        self.num_papers = num_papers

    def calculate(self):
        for feature in self.features:
            studies = feature.studies[:]
            random.shuffle(studies)



###########
# HELPERS #
###########

def fix_distances(bm):
    for f in bm.keys():
        if bm[f][f] == 0:
            bm[f][f] = Distance(bm, bm[f], bm[f])

# set up caching
try:
    MEMORY = joblib.Memory(cachedir='cache', verbose=0)
except:
    os.mkdir('cache')
    MEMORY = joblib.Memory(cachedir='cache', verbose=0)

def _prune_distance_matrix(df):
    """Returns the largest complete distance matrix included in df"""
    # for label in df.index
    pass

def _emd(image1, image2, reduction, reduction_method, blur):
    """Returns Earth Mover's Distance for image1 and image2"""
    if reduction:
        # reduce resolution of image to make problem tractable
        if reduction_method == 'block_reduce':
            image1 = _block_reduce(image1, reduction, blur)
            image2 = _block_reduce(image2, reduction, blur)
        elif reduction_method == 'spline':
            image1 = ndimage.interpolation.zoom(image1, 1./reduction)
            image2 = ndimage.interpolation.zoom(image2, 1./reduction)
        else:
            raise ValueError('No reduction method: ' + reduction_method)

    # turn voxels into probability distributions
    image1, image2 = [np.clip(img, 0, 999) for img in (image1, image2)]
    image1, image2 = [img / np.sum(img) for img in (image1, image2)]

    result = pyemd.emd(image1.ravel(), image2.ravel(), _distance_matrix(image1.shape))
    return result

def _block_reduce(image, factor, blur):
    """Returns a reduced resolution copy of given 3d image

    First, a gaussian blur is applied. Then the image is broken into
    cubes with side length of factor. The returned image is made up
    of the means of each block."""
    if image.ndim == 1:  # flattened image
        image = image.reshape(91, 109, 91)  # shape of MNI space
        reshaped = True
    else:
        reshaped = False

    image = ndimage.filters.gaussian_filter(image, blur)
    reduced = block_reduce(image, block_size=(factor, factor, factor), func=np.mean)
    if reshaped:
        reduced = reduced.flatten  # return with same format as input
    return reduced


@MEMORY.cache
def _distance_matrix(shape):
    """Returns a distance matrix for all points in a space with given shape"""
    m = np.mgrid[:shape[0], :shape[1], :shape[2]]
    coords = np.array([m[0].ravel(), m[1].ravel(), m[2].ravel()]).T
    return spatial.distance.squareform(spatial.distance.pdist(coords))


def _getdata():
    """Downloads data from neurosynth and pickles it"""
    from neurosynth.base.dataset import download
    print 'Downloding dataset'
    download(path='.', unpack=True)  # get latest data files
    data = Dataset('database.txt')
    data.add_features('features.txt')
    data.save('data/dataset.pkl')  # save in neurosynth-usable format
    os.remove('database.txt')
    os.remove('features.txt')


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt


#################
# CUSTOM SCRIPT #
#################

def jaccard(bm,  f1, f2):
    s1 = set(bm[f1].studies)
    s2 = set(bm[f2].studies)
    return len(s1 & s2) / len(s1 | s2) 



def log(txt):
    """Appends txt to log.txt"""
    with open('log.txt', 'a+') as f:
        f.write(str(txt)+'\n')


def pairs(lst):
    """Returns all possible pairs from elements in lst"""
    pairs = []
    for i, a in enumerate(lst):
        for b in lst[i+1:]:
            pairs.append((a, b))
    return pairs


def permutations(parameters):
    """Returns all possible permutations of params in parameters

    parmeters must be a list of (str, list) tuples, where str is
    the key and list, a list of values. A list of dictionaries
    is returned."""

    def recurse(parameters, permutations):
        if not parameters:
            return permutations

        # return a copy of permutations with all possible values
        #   for param added to each permutation
        # this multiplies len(permutations) by len(values)
        param, values = parameters.pop(0)
        new_perms = []
        for v in values:
            perm_copy = copy.deepcopy(permutations)
            for perm in perm_copy:
                perm[param] = v
            new_perms += perm_copy

        return recurse(parameters, new_perms)

    parameters.reverse()  # so that result is sorted by first parameter
    param, values = parameters.pop(0)
    permutations = [{param: val} for val in values]
    return recurse(parameters, permutations)


def custom_script():
    """A custom script to be run on execution"""
    try:
        #################
        bm = BrainMatrix(reduction=5)
        features = ['perception', 'visual', 'auditory', 'olfactory', 'tactile', 'somatosensory', 'language', 'language_comprehension', 'syntactic', 'semantic', 'second_language', 'spatial', 'sequential', 'social', 'music', 'memory', 'working_memory', 'eye_movement', 'arm', 'hand', 'finger', 'foot', 'speech',]
        # for f1, f2 in pairs(features):
            # bm[f1][f2]
        # bm.compute_distances()
        # bm.write_csv()
        # bm.plot_dendrogram(features)
        # bm.plot(features)
        # bm.plot(features, distance_measure='peak')
        #bm.plot_dendrogram(features, distance_measure = 'peak')
        bm.plot_dendrogram(features, distance_measure = 'emd')
        #################

    except:
        LOG.exception('Exception in script:')
        try:
            from IPython.core import ultratb
            sys.excepthook = ultratb.FormattedTB(call_pdb=1)
        except:
            pass
        else:
            raise
    #finally:
    #    if SAVE:
    #        # save all BrainMatrices
    #        all_brain_matrices = []
    #        for v in vars().values():
    #            if isinstance(v, BrainMatrix):
    #                all_brain_matrices.append(v)
    #            elif isinstance(v, list):
    #                for i in v:
    #                    if isinstance(i, BrainMatrix):
    #                        if i not in all_brain_matrices:
    #                            all_brain_matrices.append(i)
    #            elif isinstance(v, dict):
    #                for w in v.values():
    #                    if isinstance(i, BrainMatrix):
    #                        if w not in all_brain_matrices:
    #                            all_brain_matrices.append(w)

    #        for bm in all_brain_matrices:
    #            try:
    #                bm.save()
    #            except:
    #                LOG.exception('Exception during save:')


def main(args):
    if 'interact' in args:
        IPython.embed()  # interactive shell

    else:
        if not SAVE:
            print 'WARNING: SAVE is False'
        script_source = inspect.getsourcelines(custom_script)[0]
        script_source = ''.join([line[4:] for line in script_source])  # unindent
        script_source = script_source.split('#################')[1]
        log('\n\n------------------------------------')
        log('DATE: ' + time.strftime('%m/%d at %H:%M'))
        log('SCRIPT:\n' + script_source)
        log('\nOUTPUT:')
        start = time.time()
        custom_script()
        runtime = time.time() - start
        hours = int(runtime // 60 ** 2)
        minutes = int((runtime % 60 ** 2) // 60)
        seconds = int(runtime % 60)
        log('\nRUN TIME: ' + '{}:{}:{}'.format(hours, minutes, seconds))


if __name__ == '__main__':
    # import sys
    main(sys.argv[1:])
    #bm = BrainMatrix(reduction=6)
    #bm.canonical_correlation_analysis(['syntactic', 'semantic', 'anxiety', 'depression'])
