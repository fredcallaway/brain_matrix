from brain_matrix import BrainMatrix, load_brainmatrix

if __name__ == '__main__':
    matrix = BrainMatrix(downsample=30, multi=False, name='test')
    matrix.compute_distances(['syntactic', 'speech', 'semantic', 'music'], 'emd')
    del matrix
    matrix = load_brainmatrix('test')
    print(matrix['semantic']['speech']['emd'])
    # create a figure in figs/
    matrix.plot(clusters=2)
    matrix.plot_dendrogram()
    # write the matrix to a csv
    matrix.write_csv()