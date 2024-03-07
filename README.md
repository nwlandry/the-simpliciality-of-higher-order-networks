# The simpliciality of higher-order networks

This repository accompanies the article ["The simpliciality of higher-order networks"](https://doi.org/10.1140/epjds/s13688-024-00458-1) by Nicholas Landry, Jean-Gabriel Young, and Nicole Eikmeier.

### The structure of this repository is as follows:
* The `Data` folder contains all of the global characteristics of the datasets (shown in Table 1 in the text), the `Figures` folder contains PDF and PNG files for each of the figures in the paper, and the `tests` folder contains unit tests to validate the code written for generating our results.
* The `sod` (Simpliciality of Data) folder contains all of the measures of simpliciality used in the paper as well as any utility functions.

### General things:
* To run the unit tests and use the package, you need to pip install the package locally. Navigate to the local folder on your computer and run
```
pip install -e .
```
if you wish to be able to edit the distribution and
```
pip install .
```
if you don't.
* To run the unit tests, run `pytest` in the command line.
* The package is referenced as `sod` (Simpliciality of Data) when accessing the functionality.
* There are also distance versions of some simpliciality measures in the code.

### Scripts
* `draw.py` provides some additional functionality for drawing the multilayer hypergraph visualizations.
* `empirical_simpliciality.py` measures the simpliciality (all three measures) of the empirical datasets and stores the results in a JSON file in the `Data` folder.
* `generate_dcsbm_parameters.py` infers the parameters of the biSBM for a given empirical dataset for use in the model fitting script and stores as a JSON file in the `Data` folder.
* `model_fitting.py` generates realizations of the generative models, measures the resulting simpliciality, and then stores the results in a JSON file in the `Data` folder.
* `simplicial_assortativity.py` generates the empirical values of simplicial assortativity contained in Table 2.
* `setup.py` allows users to pip install this package.

### Notebooks
* `plot_empiricial_simpliciality.ipynb` generates a plot of the simpliciality for empirical datasets, which is unused in the text. It also prints 
* `plot_model_fitting.ipynb` generates Fig. 2 in the text.
* `local_simpliciality.ipynb` generates Fig. 3 in the text as well as corresponding local measures.
* `dataset_characteristics.ipynb` generates the results in Table 1 except the measures of simpliciality.
* `illustrations.ipynb` generates the diagrams used in Fig. 1 in the text.
* `simpliciality_correlation.ipynb` generates the correlation coefficients referenced in the text.
* `cm_convergence.ipynb` generates Fig. 4 in the text.
* `print_simplicial_assortativity.ipynb` prints the results from `simplicial_assortativity.py` as Table 2.
