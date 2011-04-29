#define M 36
#define N 36
#define K 36
#define alpha 1
#define beta 1
double A[M][K+13];
double B[K][N+13];
double C[M][N+13];

#include <stdio.h>

void init_array()
{
  int i, j;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      A[i][j] = (i + j);
      // We do not want to optimize this.
      __sync_synchronize();
      B[i][j] = (double)(i*j);
      C[i][j] = 0.0;
    }
  }
}


void print_array()
{
  int i, j;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      fprintf(stdout, "%lf ", C[i][j]);
      if (j%80 == 79) fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
  }
}


void do_pluto_matmult(void) {
  int i, j, k;

  __sync_synchronize();
  i = 0;
  do {
    j = 0;
    do {
      k = 0;
      do {
        C[i][j] = beta*C[i][j] + alpha*A[i][k] * B[k][j];
        ++k;
      } while (k < K);
      ++j;
    } while (j < N);
    ++i;
  } while (i < M);
  __sync_synchronize();
}

int main()
{
    register double s;

    init_array();

#pragma scop
    do_pluto_matmult();
#pragma endscop
    print_array();

  return 0;
}
