// RUN: %clangxx_asan -O0 %s -o %t && %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc];
  free(x);
  free(x + argc - 1);  // BOOM
  // CHECK: AddressSanitizer: attempting double-free{{.*}}in thread T0
  // CHECK: double-free.cc:[[@LINE-2]]
  // CHECK: freed by thread T0 here:
  // CHECK: double-free.cc:[[@LINE-5]]
  // CHECK: allocated by thread T0 here:
  // CHECK: double-free.cc:[[@LINE-10]]
  return res;
}
