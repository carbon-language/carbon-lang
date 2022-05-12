// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <malloc.h>

int main() {
  int *buffer = (int*)calloc(42, sizeof(int));
  free(buffer);
  buffer[0] = 42;
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread T0
// CHECK-NEXT: {{#0 .* main .*calloc_uaf.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 0 bytes inside of 168-byte region
// CHECK: freed by thread T0 here:
// CHECK-NEXT: {{#0 .* free }}
// CHECK-NEXT: {{#1 .* main .*calloc_uaf.cpp}}:[[@LINE-8]]
// CHECK: previously allocated by thread T0 here:
// CHECK-NEXT: {{#0 .* calloc }}
// CHECK-NEXT: {{#1 .* main .*calloc_uaf.cpp}}:[[@LINE-12]]
}
