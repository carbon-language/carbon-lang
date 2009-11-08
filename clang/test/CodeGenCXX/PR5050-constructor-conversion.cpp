// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s
// RUN: true

struct A { A(const A&, int i1 = 1); };

struct B : A { };

A f(const B &b) {
  return b;
}

// CHECK-LP64: call     __ZN1AC1ERKS_i

// CHECK-LP32: call     L__ZN1AC1ERKS_i


