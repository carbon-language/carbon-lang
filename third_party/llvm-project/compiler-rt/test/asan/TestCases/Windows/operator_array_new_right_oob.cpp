// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <windows.h>

int main() {
  char *buffer = new char[42];
  buffer[42] = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK:   {{#0 .* main .*operator_array_new_right_oob.cpp}}:[[@LINE-3]]
// CHECK: [[ADDR]] is located 0 bytes to the right of 42-byte region
// CHECK: allocated by thread T0 here:
// FIXME: The 'operator new' frame should have [].
// CHECK:   {{#0 .* operator new}}
// CHECK:   {{#1 .* main .*operator_array_new_right_oob.cpp}}:[[@LINE-9]]
  delete [] buffer;
}
