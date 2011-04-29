#include <string.h>
#define N 1024

int main () {
  int i;
  int A[N];

  memset(A, 0, sizeof(int) * N);

  for (i = 0; i < N; i++) {
    A[i] = 1;
  }

  for (i = 0; i < N; i++)
    if (A[i] != 1)
      return 1;

  return 0;
}

