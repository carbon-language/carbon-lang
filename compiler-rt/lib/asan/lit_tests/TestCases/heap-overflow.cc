// RUN: %clangxx_asan -O0 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O1 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O2 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O3 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in _?main .*heap-overflow.cc:}}[[@LINE-2]]
  // CHECK: {{0x.* is located 0 bytes to the right of 10-byte region}}
  // CHECK: {{allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*malloc}}
  // CHECK-Linux: {{    #1 0x.* in main .*heap-overflow.cc:13}}

  // CHECK-Darwin: {{    #0 0x.* in _?wrap_malloc.*}}
  // CHECK-Darwin: {{    #1 0x.* in _?main .*heap-overflow.cc:13}}
  free(x);
  return res;
}
