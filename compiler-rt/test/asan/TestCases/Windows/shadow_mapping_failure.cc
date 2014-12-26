// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>

char bigchunk[1 << 29];

int main() {
  printf("Hello, world!\n");
  scanf("%s", bigchunk);
// CHECK-NOT: Hello, world!
// CHECK: ERROR: AddressSanitizer failed to allocate
// CHECK: ReserveShadowMemoryRange failed
}
