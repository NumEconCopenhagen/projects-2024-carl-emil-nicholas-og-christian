# Data analysis project

Our project is titled **Rolling the DICE in metaanalysis** and is about determining the functional form of the damage function in William D. Nordhaus's new 2023-version of the Dynamic Integrated Climate-Economy model (DICE). We find the functional form through a metanalysis of studies from leading climate-economy experts. The datapoints from these studies have been summarized in Nordhaus' own metaanalysis as well as in Richard Tol's metaanalysis. The datapoints consists of increases in global temperature above the mean in 1920 and the corresponding loss in global GDP. We test several functional forms on the metaanalysis data through a regression analysis using OLS. We find significant results for a linear, hyperbolic sinus, quadratic and exponentiel function form. However, we choose conclude that only the quadratic and exponentiel functional forms are reasonable, since the damages has to show convexity according to leading experts such as Simon Dietz (2015).

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We apply the **following datasets**:

1. NordhausD.csv (*source*) 
1. TolD.csv (*source*)

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``
