#include <string.h>
#include <stdio.h>
#define N 1021

int main () {
  int i;
  int A[N];
  int red;

  memset(A, 0, sizeof(int) * N);

  A[0] = 1;
  A[1] = 1;
  red = 0;

  __sync_synchronize();

  for (i = 2; i < N; i++) {
    A[i] = A[i-1] + A[i-2];
    red += A[i-2];
  }

  __sync_synchronize();

  if (red != 382399368)
    return 1;
}
