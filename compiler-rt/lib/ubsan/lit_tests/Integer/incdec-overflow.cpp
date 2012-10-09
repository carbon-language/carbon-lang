// RUN: %clang -DOP=n++ -fcatch-undefined-behavior %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clang -DOP=++n -fcatch-undefined-behavior %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clang -DOP=m-- -fcatch-undefined-behavior %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clang -DOP=--m -fcatch-undefined-behavior %s -o %t && %t 2>&1 | FileCheck %s

#include <stdint.h>

int main() {
  int n = 0x7ffffffd;
  n++;
  n++;
  int m = -n - 1;
  // CHECK: incdec-overflow.cpp:15:3: fatal error: signed integer overflow: [[MINUS:-?]]214748364
  // CHECK: + [[MINUS]]1 cannot be represented in type 'int'
  OP;
}
