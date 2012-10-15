// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m64 -O1 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m64 -O2 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O1 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O2 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

#include <string.h>
int main(int argc, char **argv) {
  char a1[] = {argc, 2, 3, 4};
  char a2[] = {1, 2*argc, 3, 4};
  int res = memcmp(a1, a2, 4 + argc);  // BOOM
  // CHECK: AddressSanitizer: stack-buffer-overflow
  // CHECK: {{#0.*memcmp}}
  // CHECK: {{#1.*main}}
  return res;
}
