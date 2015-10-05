from brain_matrix import BrainMatrix, load_brainmatrix
from distance import euclidean_emd
from utils import Timer

if __name__ == '__main__':
    print('\n\nRUNNING TESTS\n----------------------------')
    with Timer('TESTS PASSED'):
        # euclidaen_emd is a function of type (img * img) -> float
        matrix = BrainMatrix(euclidean_emd, downsample=30, name='test')
        matrix.compute_distances(['syntactic', 'speech', 'semantic', 'music'])
        del matrix
        matrix = load_brainmatrix('test')
        assert isinstance(matrix['semantic']['speech'].distance, float)
        # create some figure in figs/
        matrix.plot(clusters=2, dim=3, interactive=True)
        matrix.plot_dendrogram()
        matrix.write_csv()