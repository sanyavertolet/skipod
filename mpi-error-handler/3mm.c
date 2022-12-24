/* Include benchmark-specific header. */
#include "3mm.h"

#include <mpi.h>
#include <mpi-ext.h>

#include <signal.h>

#define MAIN_PROC_RANK 0

#define EXTRALARGE_DATASET

double bench_t_start, bench_t_end;
int size, rank;

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
    bench_t_start = MPI_Wtime();
}

void bench_timer_stop() {
    bench_t_end = MPI_Wtime();
}

void bench_timer_print() {
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static void init_array(int ni, int nj, int nk, int nl, int nm, float A[ni][nk], float B[nk][nj], float C[nj][nm], float D[nm][nl]) {
    int i, j;

    for(i = 0; i < ni; i++) {
        for (j = 0; j < nk; j++) {
            A[i][j] = (float)((i * j + 1) % ni) / (5 * ni);
        }
    }

    for (i = 0; i < nk; i++) {
        for (j = 0; j < nj; j++) {
            B[i][j] = (float)((i * (j + 1) + 2) % nj) / (5 * nj);
        }
    }
  
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nm; j++) {
            C[i][j] = (float)(i * (j + 3) % nl) / (5 * nl);
        }
    }

    for (i = 0; i < nm; i++) {
        for (j = 0; j < nl; j++) {
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

MPI_Comm main_comm;

static int kernel_3mm(int ni, int nj, int nk, int nl, int nm, float E[ni][nj], float A[ni][nk], float B[nk][nj], float F[nj][nl], float C[nj][nm], float D[nm][nl], float G[ni][nl], float E_gather[ni][nj], float F_gather[nj][nl], float G_gather[ni][nl], int *rcounts, int *shifts) {
    int i, j, k;
    int i_from, i_to, j_from, j_to;

    i_from = rank * ni / size;
    i_to = (rank + 1) * ni / size;

    j_from = 0;
    j_to = nj;
    
    for (i = i_from; i < i_to; i++) {
        for (j = j_from; j < j_to; j++) {
            E[i][j] = 0.0f;
            for (k = 0; k < nk; ++k) {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    shifts[0] = 0;
    for (int i = 0; i < size; i++) {
        rcounts[i] = ((i + 1) * ni / size - i * ni / size) * nj;
        if (i != 0) {
            shifts[i] = shifts[i - 1] + rcounts[i - 1];
        }
    }

    int ret_code = MPI_Barrier(main_comm);
    if (ret_code != MPI_SUCCESS){
        return 1;
    }
    MPI_Allgatherv(&E[i_from][j_from], rcounts[rank], MPI_FLOAT, E_gather, rcounts, shifts, MPI_FLOAT, main_comm);


    i_from = rank * nj / size;
    i_to = (rank + 1) * nj / size;

    j_from = 0;
    j_to = nl;

    for (i = i_from; i < i_to; i++) {
        for (j = j_from; j < j_to; j++) {
            F[i][j] = 0.0f;
            for (k = 0; k < nm; ++k) {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }
    shifts[0] = 0;
    for (int i = 0; i < size; i++) {
        rcounts[i] = ((i + 1) * nj / size - i * nj / size) * nl;
        if (i != 0) {
            shifts[i] = shifts[i - 1] + rcounts[i - 1];
        }
    }
    ret_code = MPI_Barrier(main_comm);
    if (ret_code != MPI_SUCCESS){
        return 2;
    }
    MPI_Allgatherv(&F[i_from][j_from], rcounts[rank], MPI_FLOAT, F_gather, rcounts, shifts, MPI_FLOAT, main_comm);
    
    
    i_from = rank * ni / size;
    i_to = (rank + 1) * ni / size;

    j_from = 0;
    j_to = nl;

    for (i = i_from; i < i_to; i++) {
        for (j = j_from; j < j_to; j++) {
            G[i][j] = 0.0f;
            for (k = 0; k < nj; ++k) {
                G[i][j] += E_gather[i][k] * F_gather[k][j];
            }
        }
    }

    shifts[0] = 0;
    for (int i = 0; i < size; i++) {
        rcounts[i] = ((i + 1) * ni / size - i * ni / size) * nl;
        if (i != 0) {
            shifts[i] = shifts[i - 1] + rcounts[i - 1];
        }
    }
    ret_code = MPI_Barrier(main_comm);
    if (ret_code != MPI_SUCCESS){
        return 3;
    }
    MPI_Allgatherv(&G[i_from][j_from], rcounts[rank], MPI_FLOAT, G_gather, rcounts, shifts, MPI_FLOAT, main_comm);
    return 0;
}

static void error_handler(MPI_Comm *comm, int *err, ...) {
    int len;
    char errstr[MPI_MAX_ERROR_STRING];
    MPI_Comm_rank(main_comm, &rank);
    MPI_Comm_size(main_comm, &size);
    MPI_Error_string(*err, errstr, &len);
    printf("Rank %d / %d: notified of error %s\n", rank, size, errstr);

    MPIX_Comm_shrink(main_comm, &main_comm);
    MPI_Comm_rank(main_comm, &rank);
    MPI_Comm_size(main_comm, &size);
}

int main(int argc, char **argv) {
    int ni;
    int nj;
    int nk;
    int nl;
    int nm;
    
    int suite_size;

    if (argc < 2) {
        fprintf(stderr, "Usage: mpicc 3mm.c -o 3mm && mpiexec -np NUM_OF_THREAD ./3mm TEST_SIZE\n");
        return -1;
    } else {
        suite_size = atoi(argv[1]);
    }
    get_sizes(suite_size, &ni, &nj, &nk, &nl, &nm);

    float (*A)[ni][nk];
    float (*B)[nk][nj];
    float (*C)[nj][nm];
    float (*D)[nm][nl];
    float (*E)[ni][nj];
    float (*E_gather)[ni][nj];
    float (*F)[nj][nl];
    float (*F_gather)[nj][nl];
    float (*G)[ni][nl];
    float (*G_gather)[ni][nl];
    
    int *rcounts;
    int *shifts;

    A = malloc((ni) * (nk) * sizeof(float));
    B = malloc((nk) * (nj) * sizeof(float));
    C = malloc((nj) * (nm) * sizeof(float));
    D = malloc((nm) * (nl) * sizeof(float));
    E = malloc((ni) * (nj) * sizeof(float));
    E_gather = malloc((ni) * (nj) * sizeof(float));
    F = malloc((nj) * (nl) * sizeof(float));
    F_gather = malloc((nj) * (nl) * sizeof(float));
    G = malloc((ni) * (nl) * sizeof(float));
    G_gather = malloc((ni) * (nl) * sizeof(float));

    init_array(ni, nj, nk, nl, nm, *A, *B, *C, *D);    
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    main_comm = MPI_COMM_WORLD;

    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);
    MPI_Barrier(main_comm);


    if (rank == 2) {
        printf("Process %d died\n", rank);
        raise(SIGKILL);
    }

checkpoint:

    rcounts = malloc(size * sizeof(int));
    shifts = malloc(size * sizeof(int));
    
    bench_timer_start();

    int ret_code = kernel_3mm(ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G, *E_gather, *F_gather, *G_gather, rcounts, shifts);
    if (ret_code != 0){
        printf("%d processes alive\n", size);
        goto checkpoint;
    }

    bench_timer_stop();
    
    if (rank == MAIN_PROC_RANK) {
        bench_timer_print();
    }

    if (rank == 0 && argc > 1 && !strcmp(argv[1], "-d")) {
        print_array(ni, nl, *G_gather);
    }

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(E_gather);
    free(F);
    free(F_gather);
    free(G);
    free(G_gather);
    free(rcounts);
    free(shifts);

    MPI_Finalize();
    return 0;
}
