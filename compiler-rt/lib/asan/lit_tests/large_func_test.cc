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
__attribute__((noinline))
static void LargeFunction(int *x, int zero) {
  x[0]++;
  x[1]++;
  x[2]++;
  x[3]++;
  x[4]++;
  x[5]++;
  x[6]++;
  x[7]++;
  x[8]++;
  x[9]++;

  x[zero + 111]++;  // we should report this exact line

  x[10]++;
  x[11]++;
  x[12]++;
  x[13]++;
  x[14]++;
  x[15]++;
  x[16]++;
  x[17]++;
  x[18]++;
  x[19]++;
}

int main(int argc, char **argv) {
  int *x = new int[100];
  LargeFunction(x, argc - 1);
  delete x;
}

// CHECK: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
// CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
// CHECK: {{READ of size 4 at 0x.* thread T0}}

// atos incorrectly extracts the symbol name for the static functions on
// Darwin.
// CHECK-Linux:  {{    #0 0x.* in LargeFunction.*large_func_test.cc:32}}
// CHECK-Darwin: {{    #0 0x.* in .*LargeFunction.*large_func_test.cc:32}}

// CHECK: {{    #1 0x.* in _?main .*large_func_test.cc:48}}
// CHECK: {{0x.* is located 44 bytes to the right of 400-byte region}}
// CHECK: {{allocated by thread T0 here:}}
// CHECK: {{    #0 0x.* in operator new.*}}
// CHECK: {{    #1 0x.* in _?main .*large_func_test.cc:47}}
