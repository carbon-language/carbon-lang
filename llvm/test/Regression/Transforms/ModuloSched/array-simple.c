#include <stdio.h>

int main (int argc, char** argv) {
  int a[25];
  
  for (i=0; i < 25; i++) {
    a[i] = 24-i;
  }

  for (i=0; i < 25; i++)
    printf("a[%d] = %d\n", i, a[i]);

  return 0;
}
