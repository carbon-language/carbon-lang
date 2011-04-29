#define N 1024

float C[N], X[N];

float parallel_loop_simple2() {
  int j;

  for (j = 0; j < N; j++)
    C[j] = j;

  for (j = 0; j < N; j++)
    X[j] += X[j];

  return C[42] + X[42];
}
