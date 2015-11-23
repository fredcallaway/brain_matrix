from matplotlib import pyplot as plt
import numpy as np
import os
import sklearn
import scipy.cluster

labels_and_points = []  # a hack to get around namespace problems

def mds(df, name="", dim=2, metric=True, clustering=True, clusters=4, interactive=False):
    """Saves a scatterplot of the items projected onto 2 dimensions.

    Uses MDS to project items onto a 2 or 3 dimensional based on their
    distances from each other. If items is not None, only plot those
    that are given. If interactive is truthy, an interactive plot will pop up. This is
    recommended for 3D graphs which are hard to make sense of without
    rotating the graph.
    """
    items = df.index
    plt.clf()

    if clustering:
        clustering = sklearn.cluster.AgglomerativeClustering(
                        linkage='complete', affinity='precomputed', n_clusters=clusters)
        assignments = clustering.fit_predict(df)
    
    if dim == 2:
        mds = sklearn.manifold.MDS(n_components=2, metric=metric, eps=1e-9, dissimilarity="precomputed")
        points = mds.fit(df).embedding_
        plt.scatter(points[:,0], points[:,1], c=assignments, s=40)
        for label, x, y in zip(items, points[:, 0], points[:, 1]):
            plt.annotate(label, xy = (x, y), xytext = (-5, 5),
                         textcoords = 'offset points', ha = 'right', va = 'bottom')
    else:
        if dim is not 3:
            raise ValueError('dim must be 2 or 3. {} provided'.format(dim))
        from mpl_toolkits.mplot3d import Axes3D  # used implicitly
        from mpl_toolkits.mplot3d import proj3d
            
        mds = sklearn.manifold.MDS(n_components=3, metric=metric, eps=1e-9, dissimilarity="precomputed")
        points = mds.fit(df).embedding_
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        xs, ys, zs = np.split(points, 3, axis=1)
        ax.scatter(xs,ys,zs, c=assignments, s=40)

        # make labels move as the user rotates the graph
        global labels_and_points  # a hack for namespace problems
        labels_and_points = []
        for feature, x, y, z in zip(items, xs, ys, zs):
            x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
            label = plt.annotate(
                feature, 
                xy = (x2, y2), xytext = (-5, 5),
                textcoords = 'offset points', ha = 'right', va = 'bottom')
            labels_and_points.append((label, x, y, z))

        def update_position(e):
            for label, x, y, z in labels_and_points:
                x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
                label.xy = x2,y2
                label.update_positions(fig.canvas.renderer)
            fig.canvas.draw()

        fig.canvas.mpl_connect('motion_notify_event', update_position)

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/{}_mds{}.png'.format(name, dim))
    if interactive:
        plt.show()

    return mds.stress_


def dendrogram(df, name="", method='complete'):
    """Plots a dendrogram using hierarchical clustering. Returns inconsistency.

    See scipy.cluster.hierarchy.linkage for details regarding
    possible clustering methods.
    """
    items = df.index
    plt.clf()

    clustering = scipy.cluster.hierarchy.linkage(df, method=method)
    scipy.cluster.hierarchy.dendrogram(clustering, orientation='left', truncate_mode=None,
                                 labels=items, color_threshold=0)

    plt.tight_layout()

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/{}_dendrogram.png'.format(name))
    return scipy.cluster.hierarchy.inconsistent(clustering)


if __name__ == '__main__':
    import pandas as pd
    a,b,c = 'abc'
    df = pd.DataFrame({a:{a:0, b:1, c:9},
                       b:{a:1, b:0, c:9},
                       c:{a:9, b:9, c:0},})
    dendrogram(df, "foobar")
    mds(df, "foobar", interactive=True, clusters=2)
