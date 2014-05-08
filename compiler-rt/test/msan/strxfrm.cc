// RUN: %clangxx_msan -m64 -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const char *p = "abcdef";
  char q[10];
  size_t n = strxfrm(q, p, sizeof(q));
  assert(n < sizeof(q));
  __msan_check_mem_is_initialized(q, n + 1);
  return 0;
}
