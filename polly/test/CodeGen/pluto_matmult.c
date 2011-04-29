#define M 2048
#define N 2048
#define K 2048
#define alpha 1
#define beta 1
double A[M][K+13];
double B[K][N+13];
double C[M][N+13];

void init_array();
void print_array();

void pluto_matmult(void) {
  int i, j, k;

  __sync_synchronize();
  for(i=0; i<M; i++)
    for(j=0; j<N; j++)
      for(k=0; k<K; k++)
        C[i][j] = beta*C[i][j] + alpha*A[i][k] * B[k][j];
  __sync_synchronize();
}

int main()
{
    register double s;

    init_array();

#pragma scop
    pluto_matmult();
#pragma endscop
    print_array();

  return 0;
}
