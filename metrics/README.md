# Metrics

### Structure: 
[metrics.py](./metrics.py) contains our implementation of the metrics described in the lecture and therefore our solution to the first tasks of assignment 01.
It is mostly meant as a source file to be imported in other scripts.

[corpus_statistics.sh](./corpus_statistics.sh) is a bash script that calculates the relevant corpus statistics as described in task 01 of assignment 01. 
It uses a positional argument as the path to the corpus file and contains a global precision parameter that determines the precision of the calculation of the average sentence length. 

The [./data](./data/) directory contains the data files as well as the [stats.md](./data/stats.md) where we put the results of the corpus_statistics.sh script, 
that we ran on each data file. 

