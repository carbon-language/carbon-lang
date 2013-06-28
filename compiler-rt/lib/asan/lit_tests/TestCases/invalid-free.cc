// RUN: %clangxx_asan -O0 %s -o %t && %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc];
  free(x + 5);  // BOOM
  // CHECK: AddressSanitizer: attempting free on address{{.*}}in thread T0
  // CHECK: invalid-free.cc:[[@LINE-2]]
  // CHECK: is located 5 bytes inside of 10-byte region
  // CHECK: allocated by thread T0 here:
  // CHECK: invalid-free.cc:[[@LINE-8]]
  return res;
}
