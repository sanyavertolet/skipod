all: 3mm

debug: 3mm
	mpiexec -np 4 ./3mm 2 -d

singled: 3mm
	mpiexec -np 1 ./3mm 2 -d

run: 3mm
	mpiexec -np 4 ./3mm 2

3mm: 3mm.c
	mpicc 3mm.c -o 3mm

clean: 
	rm -f 3mm

