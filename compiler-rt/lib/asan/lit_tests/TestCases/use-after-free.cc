// RUN: %clangxx_asan -O0 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O1 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O2 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O3 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out

#include <stdlib.h>
int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*use-after-free.cc:14}}
  // CHECK: {{0x.* is located 5 bytes inside of 10-byte region .0x.*,0x.*}}
  // CHECK: {{freed by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*free}}
  // CHECK-Linux: {{    #1 0x.* in main .*use-after-free.cc:13}}

  // CHECK-Darwin: {{    #0 0x.* in wrap_free}}
  // CHECK-Darwin: {{    #1 0x.* in main .*use-after-free.cc:13}}

  // CHECK: {{previously allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*malloc}}
  // CHECK-Linux: {{    #1 0x.* in main .*use-after-free.cc:12}}

  // CHECK-Darwin: {{    #0 0x.* in wrap_malloc.*}}
  // CHECK-Darwin: {{    #1 0x.* in main .*use-after-free.cc:12}}
}
