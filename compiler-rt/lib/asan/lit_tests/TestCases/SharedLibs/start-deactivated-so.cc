#include <stdio.h>
#include <stdlib.h>

extern "C" void do_another_bad_thing() {
  char *volatile p = (char *)malloc(100);
  printf("%hhx\n", p[105]);
}
