#include "limits.h"
#define N 20

int main () {
  long long i;
  long long A[N];

  A[0] = 0;

  __sync_synchronize();

  for (i = 0; i < LLONG_MAX; i++)
    A[0] = i;

  __sync_synchronize();

  if (A[0] == LLONG_MAX - 1)
    return 0;
  else
    return 1;
}
