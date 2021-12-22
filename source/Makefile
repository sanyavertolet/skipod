all: 3mm

run: 3mm
	./3mm 2

debug: 3mm
	./3mm 2 -d

3mm: 3mm.o
	gcc 3mm.o -o 3mm

3mm.o: 3mm.c
	gcc 3mm.c -c

clean: 
	rm -f 3mm.o
	rm -f 3mm

