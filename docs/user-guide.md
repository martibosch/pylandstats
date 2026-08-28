# User guide

A tutorial and thorough overview of PyLandStats. The data preprocessing pipeline
that derives the example datasets used below is kept in the
[pylandstats-notebooks](https://github.com/martibosch/pylandstats-notebooks)
repository.

```{toctree}
---
maxdepth: 1
---
user-guide/overview
user-guide/landscape-analysis
user-guide/spatiotemporal-analysis
user-guide/zonal-analysis
user-guide/spatiotemporal-zonal-analysis
user-guide/spatial-signature-analysis
```

## Annex

The FRAGSTATS comparison below is executed when building these docs, i.e., it checks
that the computed metrics match the FRAGSTATS reference values. The other annexes are
rendered from their stored outputs, since they either require data that is not shipped
with the docs or belong to the preprocessing pipeline of the
[pylandstats-notebooks](https://github.com/martibosch/pylandstats-notebooks) repository.

```{toctree}
---
maxdepth: 1
---
user-guide/a01-fragstats-comparison
user-guide/a02-performance-notes
user-guide/a03-swisslandstats-preprocessing
user-guide/a04-elevation-zones
user-guide/a05-pylandstats-3-benchmark
user-guide/a06-bird-richness-preprocessing
```
