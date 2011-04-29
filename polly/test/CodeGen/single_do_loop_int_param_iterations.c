#define N 20
#include "limits.h"
volatile int A[N];

void bar (int n) {
  int i;
  __sync_synchronize();
  i = 0;

  do {
    A[0] = i;
    ++i;
  } while (i < 2 * n);
  __sync_synchronize();
}

int main () {
  A[0] = 0;
  bar (N/2);

  if (A[0] == N - 1 )
    return 0;
  else
    return 1;
}
