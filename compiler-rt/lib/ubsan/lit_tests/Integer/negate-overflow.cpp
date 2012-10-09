// RUN: %clang -fcatch-undefined-behavior %s -o %t && %t 2>&1 | FileCheck %s

int main() {
  -unsigned(-0x7fffffff - 1); // ok
  // CHECK: negate-overflow.cpp:6:10: fatal error: negation of -2147483648 cannot be represented in type 'int'; cast to an unsigned type to negate this value to itself
  return -(-0x7fffffff - 1);
}
