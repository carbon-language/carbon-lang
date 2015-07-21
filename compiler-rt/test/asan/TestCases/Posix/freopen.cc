// RUN: %clangxx_asan -O0 %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>

int main() {
  FILE *fp = fopen("/dev/null", "w");
  assert(fp);
  freopen(NULL, "a", fp);
  fclose(fp);
  return 0;
}
