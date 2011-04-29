#define N 20
#include "limits.h"

int main () {
  int i;
  int A[N];

  A[0] = 1;

  __sync_synchronize();

  i = 0;

  do {
    A[0] = 0;
    ++i;
  } while (i < 1);

  __sync_synchronize();

  if (A[0] == 0)
    return 0;
  else
    return 1;
}
