# RAxMLpy

## Introduction

**raxmlpy** is a Python library that simplifies and automates the use of [RAxML](https://cme.h-its.org/exelixis/web/software/raxml/) (Randomized Axelerated Maximum Likelihood) for phylogenetic analyses. It provides a user-friendly Python interface to RAxML.


## Installation

To install `raxmlpy`, use the following command:

```bash
python setup.py build
python setup.py install
```


## Dynamic Libraries

The `build_plllibs` directory contains the necessary C dynamic libraries. To ensure that Python properly loads these libraries, you need to add this directory to your `LD_LIBRARY_PATH` environment variable as shown above.


```bash
export LD_LIBRARY_PATH="$(pwd)/build_plllibs:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$(pwd)/build_raxmllib:$LD_LIBRARY_PATH"
```


## Usage

After installing `raxmlpy`, you can import and use it in your Python code.

```python
import raxmlpy
# Your code using raxmlpy package...
```

## Test file

```bash
cd test
python test_raxmlpy.py
```
