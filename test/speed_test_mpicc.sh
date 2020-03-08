#!/bin/sh

make clean
make speed_test.mpi
mpiexec -np 2 ./speed_test.mpi 2 2

