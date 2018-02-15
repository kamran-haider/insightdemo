insightdemo
===========

In order to explore this demonstration independently, please download or clone the repository:
```bash
git clone git@github.com:kamran-haider/insightdemo.git
cd insightdemo
```
run the following commands:

```bash
conda env create -f environment.yml -n demo
soure activate demo
jupyter notebook
```

This is tested on macos and linux, using `Anaconda3` python distribution.

## Water displacement in structure-guided drug design - introductory data exploration and analysis

### Part 1 
Part 1 of this demonstration covers visualization of protein structures and molecular dynamics trajectories in a Jupyter notebook. 
This provides an overview of the kind of data that is involved in computer-aided structure-based drug design. We also get to see
how SSTMap reduces a complex 3-D distribution of water molecules in the active-site into a group of clusters occupying the same positions
as ligand atoms. Furthermore, it calculates thermodynamic and structural quantities of each cluster. This results in about 11 features for each 
data point.


### Part 2 
In Part 2, we see if we can train a model on a data set consisting of clusters obtained as outlined above and predict if a given
cluster is displaceable or not. We see that a simple Logistics Regression model does not do a good job in learning the highly complex displacement function.
On the other hand, we start to see improvement with a deep learning neural network. This our some very basic demonstrations and for more
detailed results, additional analysis is required. 
 