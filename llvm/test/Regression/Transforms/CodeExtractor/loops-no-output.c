#include <stdio.h>

int main(int argc, char *argv[]) {
  int i;
  for (i=0; i < 5; ++i)
    printf("%d\n", i);

  printf("separator!\n");
  
  for (i=0; i < 4; ++i)
    printf("[%d]\n", i+5);
  return 0;
}
