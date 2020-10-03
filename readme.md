# mob2vec

mob2vec is a novel framework for the representation learning of
symbolic trajectories, in particular of CDR (Call Detail Record)
trajectories. The methodology is centered on the combined use of a
recent symbolic trajectory segmentation method for the removal of noise,
a novel trajectory generalization method incorporating behavioral
information, and an unsupervised technique for the learning of vector
representations from sequential data.
 
mob2vec is the result of an empirical study 
 conducted on real CDR data through an extensive experimentation. 
 As a result, it is shown that mob2vec generates vector representations of CDR 
 trajectories in low dimensional spaces which preserve the similarity of the 
 mobility behavior of individuals.
 
 
 This research is presented in the paper "*Maria Luisa Damiani, Andrea
Acquaviva, Fatima Hachem, Matteo Rossini*. ***Learning Behavioral Representations of Human
Mobility***. *In: Proceedings of ACM SIGSPATIAL 2020, 3-6 Nov, Seattle, US. DOI: 10.1145/3397536.3422255"*

## Code
The code is divided into several executable files to facilitate their use and reading.
Executables need no arguments, only parameters in the config.json.

<br />

### #0 Data preparation
It aims to create the sequences of symbols for each week and each user, starting from our dataset.

<br />

### #1 Data preprocessing
Search for frequent sequence patterns in sequences, writing data for the training phase.

<br />

### #2 Training
Train two doc2vec models, as thought in sqn2vec.
Optionally, trained embeddings can be saved.

<br />

### #3 Embedding export
Export embeddings related to input sequences to file, using the inference method.

<br />

### #4 Dimensionality reduction
Reduces the dimensionality of embeddings with UMAP (128D -> 2D). It supports multiple values of n_neighbors and min_dist.

<br />

### #5 Evaluation with jensen-shannon divergence
For a defined number of user pairs, it correlates the jensen shannon divergence of the original rank sequences and the 
Euclidean distance of the 2D embeddings. Correlation is measured with Pearson's correlation coefficient (r).

<br />

### #6 Validation
Test suite divided into three sequential parts. The first consists in the creation of modified sequences starting 
from the original ones, removing one or more less significant symbols. The second is to export the embeddings of the 
new sequences via inference. The third shows the position of the modified sequences with respect to the original ones:
 this classification is made by taking the Euclidean distance of the relative embeddings. The script also shows the 
 ranking obtained with the edit distance for comparison purposes.
 
<br />

### #7 Plot 2D embedding dataset
Simply plot 2D embedding dataset, with the ability to juxtapose specific user labels.

<br />

### #8 Plot distance between similar trajectories
Create some boxplots to highlight the Euclidean distance between the representations of the original sequences and
 those of the sequences without a certain number of less significant symbols.

<br />

## Disclaimer
This code is provided for reading purposes and is no longer mantained.

This code was not designed for performance or application purposes, but for research purposes only.

The code not already contained in sqn2vec is distributed under the GNU LGPLv3 license.

The dataset is not available as under non-disclosure agreement.
