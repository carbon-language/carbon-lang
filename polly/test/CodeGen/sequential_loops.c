#include <string.h>
#define N 1024

int A[N];

void sequential_loops() {
  int i;
  for (i = 0; i < N/2; i++) {
    A[i] = 1;
  }
  for (i = N/2 ; i < N; i++) {
    A[i] = 2;
  }
}

int main () {
  int i;
  memset(A, 0, sizeof(int) * N);

  sequential_loops();

  for (i = 0; i < N; i++) {
    if (A[i] != 1 && i < N/2)
      return 1;
    if (A[i] !=  2 && i >= N/2)
      return 1;
  }

  return 0;
}

