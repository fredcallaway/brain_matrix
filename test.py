from brain_matrix import BrainMatrix, load_brainmatrix
from utils import Timer

if __name__ == '__main__':
    print('\n\nRUNNING TESTS\n----------------------------')
    with Timer('RUN TIME'):
        # euclidaen_emd 
        matrix = BrainMatrix('emd', downsample=10, name='test')
        print('Compute distances')
        matrix.compute_distances(['syntactic', 'speech', 'semantic', 'music'])
        del matrix
        matrix = load_brainmatrix('test')
        assert isinstance(matrix['semantic']['speech'].distance, float)
        # create some figure in figs/
        matrix.plot_mds(clusters=2, dim=3, interactive=False)
        matrix.plot_dendrogram()
        matrix.write_csv()
        print('TESTS PASSED')