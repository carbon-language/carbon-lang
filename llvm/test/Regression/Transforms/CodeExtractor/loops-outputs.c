#include <stdio.h>

int main(int argc, char *argv[]) {
  int i, d=0;
  for (i=0; i < 10; ++i)
    d += i;

  printf("separator!\n");
  
  for (i=0; i < 4; ++i)
    printf("[%d]\n", d+i);
  return 0;
}
