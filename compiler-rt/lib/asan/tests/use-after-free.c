#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}

// CHECK: heap-use-after-free
// CHECKSLEEP: Sleeping for 1 second
