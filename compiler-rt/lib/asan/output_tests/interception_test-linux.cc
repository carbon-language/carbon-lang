#include <stdlib.h>
#include <stdio.h>

extern "C" long __interceptor_strtol(const char *nptr, char **endptr, int base);
extern "C" long strtol(const char *nptr, char **endptr, int base) {
  fprintf(stderr, "my_strtol_interceptor\n");
  return __interceptor_strtol(nptr, endptr, base);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
}

// Check-Common: my_strtol_interceptor
// Check-Common: heap-use-after-free

