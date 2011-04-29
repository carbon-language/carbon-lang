#include <string.h>
#include <stdio.h>
#define N 1021

int main () {
  int i;
  int A[N];
  int RED[1];

  memset(A, 0, sizeof(int) * N);

  A[0] = 1;
  A[1] = 1;
  RED[0] = 0;

  for (i = 2; i < N; i++) {
    A[i] = A[i-1] + A[i-2];
    RED[0] += A[i-2];
  }

  if (RED[0] != 382399368)
    return 1;
}
