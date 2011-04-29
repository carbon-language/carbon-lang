#include <string.h>
int A[1];

void constant_condition () {
  int a = 0;
  int b = 0;

  if (a == b)
    A[0] = 0;
  else
    A[0] = 1;
}

int main () {
  int i;

  A[0] = 2;

  constant_condition();

  return A[0];
}

