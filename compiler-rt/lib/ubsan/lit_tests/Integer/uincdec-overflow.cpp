// RUN: %clangxx -DOP=n++ -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck --check-prefix=CHECK-INC %s
// RUN: %clangxx -DOP=++n -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck --check-prefix=CHECK-INC %s
// RUN: %clangxx -DOP=m-- -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck --check-prefix=CHECK-DEC %s
// RUN: %clangxx -DOP=--m -fsanitize=unsigned-integer-overflow %s -o %t && %t 2>&1 | FileCheck --check-prefix=CHECK-DEC %s

#include <stdint.h>

int main() {
  unsigned n = 0xfffffffd;
  n++;
  n++;
  unsigned m = 0;
  // CHECK-INC: uincdec-overflow.cpp:15:3: runtime error: unsigned integer overflow: 4294967295 + 1 cannot be represented in type 'unsigned int'
  // CHECK-DEC: uincdec-overflow.cpp:15:3: runtime error: unsigned integer overflow: 0 - 1 cannot be represented in type 'unsigned int'
  OP;
}
