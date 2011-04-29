#define N 20
#include "limits.h"
#include <stdio.h>
volatile  int A[2 * N];

void single_do_loop_scev_replace() {
  int i;

  __sync_synchronize();

  i = 0;

  do {
    A[2 * i] = i;
    ++i;
  } while (i < N);

  __sync_synchronize();
}

int main () {
  int i;

  single_do_loop_scev_replace();

  fprintf(stdout, "Output %d\n", A[0]);

  if (A[2 * N - 2] == N - 1)
    return 0;
  else
    return 1;
}
