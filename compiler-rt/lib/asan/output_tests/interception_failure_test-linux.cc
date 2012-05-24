#include <stdlib.h>
#include <stdio.h>

extern "C" long strtol(const char *nptr, char **endptr, int base) {
  fprintf(stderr, "my_strtol_interceptor\n");
  return 0;
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
}

// Check-Common: my_strtol_interceptor
// CHECK-NOT: heap-use-after-free

