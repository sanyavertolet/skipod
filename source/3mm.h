#ifndef _3MM_H
#define _3MM_H 

void get_sizes(int size, int *ni, int *nj, int *nk, int *nl, int *nm) {
    switch(size) {
        case 0:
            *ni = 16;
            *nj = 18;
            *nk = 20;
            *nl = 22;
            *nm = 24;
            break;
        case 1:
            *ni = 40;
            *nj = 50;
            *nk = 60;
            *nl = 70;
            *nm = 80;
            break;
        case 2:
            *ni = 180;
            *nj = 190;
            *nk = 200;
            *nl = 210;
            *nm = 220;
            break;
        case 3:
            *ni = 800;
            *nj = 900;
            *nk = 1000;
            *nl = 1100;
            *nm = 1200;
            break;
        case 4:
            *ni = 1600;
            *nj = 1800;
            *nk = 2000;
            *nl = 2200;
            *nm = 2400;
            break;
    }
}

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#endif
