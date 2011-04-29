#define M 1024
#define N 1024
#define K 1024

float X[K];

float parallel_loop_simple() {
  int i, k;

  for (i = 0; i < M; i++)
    for (k = 0; k < K; k++)
      X[k] += X[k];

  return X[42];
}
