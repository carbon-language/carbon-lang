// RUN: %clangxx_asan -m64 -O0 -fsanitize-address-zero-base-shadow -fPIE -pie %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-64 < %t.out
// RUN: %clangxx_asan -m64 -O1 -fsanitize-address-zero-base-shadow -fPIE -pie %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-64 < %t.out
// RUN: %clangxx_asan -m64 -O2 -fsanitize-address-zero-base-shadow -fPIE -pie %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-64 < %t.out
// RUN: %clangxx_asan -m32 -O0 -fsanitize-address-zero-base-shadow -fPIE -pie %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-32 < %t.out
// RUN: %clangxx_asan -m32 -O1 -fsanitize-address-zero-base-shadow -fPIE -pie %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-32 < %t.out
// RUN: %clangxx_asan -m32 -O2 -fsanitize-address-zero-base-shadow -fPIE -pie %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-32 < %t.out

#include <string.h>
int main(int argc, char **argv) {
  char x[10];
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in _?main .*zero-base-shadow.cc:}}[[@LINE-2]]
  // CHECK: {{Address 0x.* is .* frame}}
  // CHECK: main

  // Check that shadow for stack memory occupies lower part of address space.
  // CHECK-64: =>0x0f
  // CHECK-32: =>0x1
  return res;
}
