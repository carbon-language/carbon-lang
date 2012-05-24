#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

extern "C" void *__xsan_malloc(size_t size);
extern "C" void *malloc(size_t size) {
  write(2, "malloc call\n", sizeof("malloc call\n") - 1);
  return __xsan_malloc(size);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
}

// Check-Common: malloc call
// Check-Common: heap-use-after-free

