// RUN: %clang -fcatch-undefined-behavior -DDIVIDEND=0 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clang -fcatch-undefined-behavior -DDIVIDEND=1U %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clang -fcatch-undefined-behavior -DDIVIDEND=1.5 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clang -fcatch-undefined-behavior -DDIVIDEND='__int128(123)' %s -o %t && %t 2>&1 | FileCheck %s

int main() {
  // CHECK: div-zero.cpp:8:12: fatal error: division by zero
  DIVIDEND / 0;
}
