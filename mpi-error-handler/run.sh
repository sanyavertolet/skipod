#!/bin/zsh

source dockervars.sh
make
mpirun -np 10 --oversubscribe --with-ft ulfm ./3mm 1
