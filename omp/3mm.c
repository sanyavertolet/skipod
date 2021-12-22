/* Include benchmark-specific header. */
#include "3mm.h"

#include <omp.h>

double bench_t_start, bench_t_end;

static double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) {
        printf ("Error return from gettimeofday: %d", stat);
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
    bench_t_start = omp_get_wtime();
}

void bench_timer_stop() {
    bench_t_end = omp_get_wtime();
}

void bench_timer_print() {
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static void init_array(int ni, int nj, int nk, int nl, int nm, float A[ni][nk], float B[nk][nj], float C[nj][nm], float D[nm][nl]) {
    for(int i = 0; i < ni; ++i) {
        for(int j = 0; j < nk; ++j) {
            A[i][j] = (float)((i * j + 1) % ni) / (5 * ni);
        }
    }

    for (int i = 0; i < nk; i++) {
        for (int j = 0; j < nj; j++) {
            B[i][j] = (float)((i * (j + 1) + 2) % nj) / (5 * nj);
        }
    }
  
    for (int i = 0; i < nj; i++) {
        for (int j = 0; j < nm; j++) {
            C[i][j] = (float)(i * (j + 3) % nl) / (5 * nl);
        }
    }

    for (int i = 0; i < nm; i++) {
        for (int j = 0; j < nl; j++) {
            D[i][j] = (float)((i * (j + 2) + 2) % nk) / (5 * nk);
        }
    }
}

static void print_array(int ni, int nl, float G[ni][nl]) {
    int i, j;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s\n", "G");
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            fprintf (stderr, "%0.2f ", G[i][j]);
        }
        fprintf (stderr, "\n");
    }
    fprintf(stderr, "\nend   dump: %s\n", "G");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_3mm(int ni, int nj, int nk, int nl, int nm, float E[ni][nj], float A[ni][nk], float B[nk][nj], float F[nj][nl], float C[nj][nm], float D[nm][nl], float G[ni][nl]) {
    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            E[i][j] = 0.0f;
            for (k = 0; k < nk; ++k) {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            F[i][j] = 0.0f;
            for (k = 0; k < nm; ++k) {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }

    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            G[i][j] = 0.0f;
            for (k = 0; k < nj; ++k) {
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int ni;
    int nj;
    int nk;
    int nl;
    int nm;

    int num_threads;
    int suite_size;

    if (argc < 3) {
        fprintf(stderr, "Usage: ./3mm NUM_OF_THREADS TEST_SIZE\n");
        return -1;
    } else {
        num_threads = atoi(argv[1]);
        suite_size = atoi(argv[2]);
    }

    get_sizes(suite_size, &ni, &nj, &nk, &nl, &nm);

    omp_set_num_threads(num_threads);
    
    float (*A)[ni][nk];
    float (*B)[nk][nj];
    float (*C)[nj][nm];
    float (*D)[nm][nl];
    float (*E)[ni][nj];
    float (*F)[nj][nl];
    float (*G)[ni][nl];
    
    A = (float(*)[ni][nk])malloc ((ni) * (nk) * sizeof(float));
    B = (float(*)[nk][nj])malloc ((nk) * (nj) * sizeof(float));
    C = (float(*)[nj][nm])malloc ((nj) * (nm) * sizeof(float));
    D = (float(*)[nm][nl])malloc ((nm) * (nl) * sizeof(float));
    E = (float(*)[ni][nj])malloc ((ni) * (nj) * sizeof(float));
    G = (float(*)[ni][nl])malloc ((ni) * (nl) * sizeof(float));
    F = (float(*)[nj][nl])malloc ((nj) * (nl) * sizeof(float));

    init_array (ni, nj, nk, nl, nm, *A, *B, *C, *D);

    bench_timer_start();

    kernel_3mm (ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G);

    bench_timer_stop();
    bench_timer_print();

    if (argc == 4 && !strcmp(argv[1], "-d")) {
        print_array(ni, nl, *G);
    }

    free((void*)E);
    free((void*)A);
    free((void*)B);
    free((void*)F);
    free((void*)C);
    free((void*)D);
    free((void*)G);

  return 0;
}

