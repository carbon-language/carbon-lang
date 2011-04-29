#include "limits.h"
#define N 20

int main () {
  unsigned long long i;
  unsigned long long A[N];

  A[0] = 0;

  __sync_synchronize();

  for (i = 0; i < ULLONG_MAX; i++)
    A[0] = i;

  __sync_synchronize();

  if (A[0] == ULLONG_MAX - 1)
    return 0;
  else
    return 1;
}
