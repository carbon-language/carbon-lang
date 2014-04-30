// RUN: %clangxx -fsanitize=signed-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECKS
// RUN: %clangxx -fsanitize=unsigned-integer-overflow %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECKU

int main() {
  // CHECKS-NOT: runtime error
  // CHECKU: negate-overflow.cpp:[[@LINE+2]]:3: runtime error: negation of 2147483648 cannot be represented in type 'unsigned int'
  // CHECKU-NOT: cast to an unsigned
  -unsigned(-0x7fffffff - 1); // ok
  // CHECKS: negate-overflow.cpp:[[@LINE+2]]:10: runtime error: negation of -2147483648 cannot be represented in type 'int'; cast to an unsigned type to negate this value to itself
  // CHECKU-NOT: runtime error
  return -(-0x7fffffff - 1);
}
