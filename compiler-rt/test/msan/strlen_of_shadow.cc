// RUN: %clangxx_msan -O0 %s -o %t && %run %t

// Check that strlen() and similar intercepted functions can be called on shadow
// memory.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

const char *mem_to_shadow(const char *p) {
#if defined(__x86_64__)
  return (char *)((uintptr_t)p & ~0x400000000000ULL);
#elif defined (__mips64)
  return (char *)((uintptr_t)p & ~0x4000000000ULL);
#endif
}

int main(void) {
  const char *s = "abcdef";
  assert(strlen(s) == 6);
  assert(strlen(mem_to_shadow(s)) == 0);

  char *t = new char[42];
  t[41] = 0;
  assert(strlen(mem_to_shadow(t)) == 41);
  return 0;
}
