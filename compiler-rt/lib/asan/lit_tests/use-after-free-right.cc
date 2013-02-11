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

// Test use-after-free report in the case when access is at the right border of
//  the allocation.

#include <stdlib.h>
int main() {
  volatile char *x = (char*)malloc(sizeof(char));
  free((void*)x);
  *x = 42;
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK: {{WRITE of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in _?main .*use-after-free-right.cc:25}}
  // CHECK: {{0x.* is located 0 bytes inside of 1-byte region .0x.*,0x.*}}
  // CHECK: {{freed by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*free}}
  // CHECK-Linux: {{    #1 0x.* in main .*use-after-free-right.cc:24}}

  // CHECK-Darwin: {{    #0 0x.* in _?wrap_free}}
  // CHECK-Darwin: {{    #1 0x.* in _?main .*use-after-free-right.cc:24}}

  // CHECK: {{previously allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*malloc}}
  // CHECK-Linux: {{    #1 0x.* in main .*use-after-free-right.cc:23}}

  // CHECK-Darwin: {{    #0 0x.* in _?wrap_malloc.*}}
  // CHECK-Darwin: {{    #1 0x.* in _?main .*use-after-free-right.cc:23}}
}
