// RUN: %clangxx_asan -O0 %s -o %t && %t

// Regression test for PR17138.

#include <assert.h>
#include <string.h>

int main() {
  char buf[1024];
  char *res = (char *)strerror_r(300, buf, sizeof(buf));
  assert(res != 0);
  return 0;
}
