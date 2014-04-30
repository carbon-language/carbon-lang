// RUN: %clangxx_msan -m64 -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <sys/timeb.h>

#include <sanitizer/msan_interface.h>

int main(void) {
  struct timeb tb;
  int res = ftime(&tb);
  assert(!res);
  assert(__msan_test_shadow(&tb, sizeof(tb)) == -1);
  return 0;
}
