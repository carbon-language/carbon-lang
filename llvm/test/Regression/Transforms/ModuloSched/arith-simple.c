#include <stdio.h>

int main (int argc, char** argv) {
  int a, b, c, d, i;
  
  a = b = c = d = 1;
  
  for (i=0; i < 15; i++) {
    a = b + c;
    c = d - b;
    d = a + b;
    b = c + i;
  }

  printf("a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);

  return 0;
}
