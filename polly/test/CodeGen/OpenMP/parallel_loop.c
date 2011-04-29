#define M 1024
#define N 1024
#define K 1024

float A[M][K], B[K][N], C[M][N], X[K];

float parallel_loop() {
  int i, j, k;

  for (i = 0; i < M; i++)
    for (j = 0; j< N; j++)
      for (k = 0; k < K; k++)
        C[i][j] += A[i][k] * B[k][j];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < K; k++)
        X[k] += X[k];

  return C[42][42] + X[42];
}
