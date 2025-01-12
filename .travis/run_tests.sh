#!/bin/bash

set -xe

# Build scipp
mkdir -p build
mkdir -p install
cd build
cmake -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} $@ -DCMAKE_INSTALL_PREFIX=../install ..
make -j2 install all-tests

# Units tests
./units/test/scipp-units-test

# Core tests
./core/test/scipp-core-test

# Neutron tests
./neutron/test/scipp-neutron-test

# Python tests
python3 -m pip install -r ../python/requirements.txt
export PYTHONPATH=$PYTHONPATH:../install
cd ../python
python3 -m pytest
