// RUN: %clangxx_msan -m64 -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdint.h>
#include <sanitizer/msan_interface.h>

int main(void) {
  unsigned long long x = 0; // For 8-byte alignment.
  uint32_t x_s = 0x12345678U;
  __msan_partial_poison(&x, &x_s, sizeof(x_s));
  __msan_print_shadow(&x, sizeof(x_s));
  return 0;
}

// CHECK: Shadow map of [{{.*}}), 4 bytes:
// CHECK: 0x{{.*}}: 87654321 ........ ........ ........
