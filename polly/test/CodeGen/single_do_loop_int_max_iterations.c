#define N 20
#include "limits.h"
#include <stdio.h>
volatile  int A[N];

void single_do_loop_int_max_iterations() {
  int i;

  __sync_synchronize();

  i = 0;

  do {
    A[0] = i;
    ++i;
  } while (i < INT_MAX);

  __sync_synchronize();
}

int main () {
  int i;

  A[0] = 0;

  single_do_loop_int_max_iterations();

  fprintf(stdout, "Output %d\n", A[0]);

  if (A[0] == INT_MAX - 1)
    return 0;
  else
    return 1;
}
