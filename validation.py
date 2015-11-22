from brain_matrix import BrainMatrix
from utils import Timer

import pandas as pd
#import seaborn as sns

if __name__ == '__main__':
    features = ['language',
                'music',
                'navigation',
                'spatial',
                'reading',
                'action',
                'hippocampus',
                'auditory',
                'broca',
                'gestures',
                'hearing',
                'learning',
                'motor',
                'movement',
                'speech',
                'syntactic',
                'verbal',
                'visuospatial']
    folds = [features[a:b] for a, b in [(0,6), (6,12), (12,18)]]
    data = []
    with open('times.txt', 'w+') as log:
        for fold_id, fold in enumerate(folds):
            for ds in range(6,11):
                with Timer('DOWNSAMPLED BY {ds}\n'.format_map(locals()),
                           print_func=log.write) as t:
                    matrix = BrainMatrix('emd', downsample=ds, name='validation')
                    matrix.compute_distances(fold)
                    inconsistency = matrix.plot_dendrogram()
                    stress = matrix.plot_mds(metric=False)
                    data.append([ds, fold_id, t.elapsed, inconsistency, stress])

    df = pd.DataFrame(columns=['downsample', 'fold', 'time', 'inconsistency', 'stress'], data=data)
    df.to_pickle('validation.pkl')

    #f, ax = sns.plt.subplots(figsize=(7, 7))
    #ax.set(yscale="log")
    #sns.pointplot(x='features', y='time', hue='downsample', data=df)
    #sns.plt.show()
