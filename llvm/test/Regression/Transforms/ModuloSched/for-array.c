#include <stdio.h>

int main (int argc, char** argv) {
  int i, a[25];
  a[0] = 1;
  
  for (i=1; i < 24; i++) {
    a[i-1] += i;
    a[i] = 5;
    a[i+1] = a[i] + a[i-1];
  }

  for (i=0; i < 25; i++)
    printf("a[%d] = %d\n", i, a[i]);

  return 0;
}
