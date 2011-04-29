int bar1();
int bar2();
int bar3();
int k;
#define N 100
int A[N];

int foo (int z) {
  int i, j;

  for (i = 0; i < N; i++) {
    A[i] = i;

      for (j = 0; j < N * 2; j++)
        A[i] = j * A[i];
  }

  return A[z];
}
