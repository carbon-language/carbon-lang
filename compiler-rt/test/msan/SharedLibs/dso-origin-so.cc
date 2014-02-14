#include <stdlib.h>

#include "dso-origin.h"

void my_access(int *p) {
  volatile int tmp;
  // Force initialize-ness check.
  if (*p)
    tmp = 1;
}

void *my_alloc(unsigned sz) {
  return malloc(sz);
}
