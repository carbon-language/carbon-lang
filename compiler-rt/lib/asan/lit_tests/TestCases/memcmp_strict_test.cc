// RUN: %clangxx_asan -O0 %s -o %t && ASAN_OPTIONS=strict_memcmp=0 %t 2>&1 | FileCheck %s --check-prefix=CHECK-nonstrict
// RUN: %clangxx_asan -O0 %s -o %t && ASAN_OPTIONS=strict_memcmp=1 %t 2>&1 | FileCheck %s --check-prefix=CHECK-strict
// Default to strict_memcmp=1.
// RUN: %clangxx_asan -O0 %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=CHECK-strict

#include <stdio.h>
#include <string.h>
int main() {
  char kFoo[] = "foo";
  char kFubar[] = "fubar";
  int res = memcmp(kFoo, kFubar, strlen(kFubar));
  printf("res: %d\n", res);
  // CHECK-nonstrict: {{res: -1}}
  // CHECK-strict: AddressSanitizer: stack-buffer-overflow
  return 0;
}
