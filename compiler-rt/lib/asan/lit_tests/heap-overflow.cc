// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m64 -O1 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m64 -O2 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m64 -O3 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O0 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O1 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O2 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O3 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in _?main .*heap-overflow.cc:23}}
  // CHECK: {{0x.* is located 0 bytes to the right of 10-byte region}}
  // CHECK: {{allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*malloc}}
  // CHECK-Linux: {{    #1 0x.* in main .*heap-overflow.cc:21}}

  // CHECK-Darwin: {{    #0 0x.* in .*mz_malloc.*}}
  // CHECK-Darwin: {{    #1 0x.* in malloc_zone_malloc.*}}
  // CHECK-Darwin: {{    #2 0x.* in malloc.*}}
  // CHECK-Darwin: {{    #3 0x.* in _?main .*heap-overflow.cc:21}}
  free(x);
  return res;
}
