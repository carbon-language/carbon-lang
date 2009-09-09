// RUN: clang-cc -triple x86_64-apple-darwin -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s &&
// RUN: true

extern "C" int printf(...);

struct S {
  S() { printf("S::S()\n"); }
  int iS;
};

struct M {
  S ARR_S; 
};

int main() {
  M m1;
}

// CHECK-LP64: call __ZN1SC1Ev

// CHECK-LP32: call L__ZN1SC1Ev
