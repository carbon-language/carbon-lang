// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=183

// RUN: %clangxx_asan -O2 %s -o %t
// RUN: not %t 12 2>&1 | FileCheck %s
// RUN: not %t 100 2>&1 | FileCheck %s
// RUN: not %t 10000 2>&1 | FileCheck %s

#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  int *x = new int[5];
  memset(x, 0, sizeof(x[0]) * 5);
  int index = atoi(argv[1]);
  int res = x[index];
  // CHECK: AddressSanitizer: {{(heap-buffer-overflow|SEGV)}}
  // CHECK: #0 0x{{.*}} in main {{.*}}heap-overflow-large.cc:[[@LINE-2]]
  // CHECK: AddressSanitizer can not {{(provide additional info|describe address in more detail \(wild memory access suspected\))}}
  // CHECK: SUMMARY: AddressSanitizer: {{(heap-buffer-overflow|SEGV)}}
  delete[] x;
  return res;
}
