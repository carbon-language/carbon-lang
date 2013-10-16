// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %t      2>&1 | FileCheck %s
// RUN: not %t heap 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=poison_partial=0 %t
// RUN: ASAN_OPTIONS=poison_partial=0 %t heap
#include <string.h>
char g[21];
char *x;

int main(int argc, char **argv) {
  if (argc >= 2)
    x = new char[21];
  else
    x = &g[0];
  memset(x, 0, 21);
  int *y = (int*)x;
  return y[5];
}
// CHECK: 0 bytes to the right
