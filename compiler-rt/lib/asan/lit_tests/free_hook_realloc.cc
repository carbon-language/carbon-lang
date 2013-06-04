// Check that free hook doesn't conflict with Realloc.
// RUN: %clangxx_asan -O2 %s -o %t && %t
#include <assert.h>
#include <stdlib.h>

extern "C" {
void __asan_free_hook(void *ptr) {
  *(int*)ptr = 0;
}
}

int main() {
  int *x = (int*)malloc(100);
  x[0] = 42;
  int *y = (int*)realloc(x, 200);
  assert(y[0] == 42);
  return 0;
}
