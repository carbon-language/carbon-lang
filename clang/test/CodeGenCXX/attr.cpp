// RUN: clang-cc -triple x86_64-apple-darwin -O0 -S %s -o %t.s &&
// RUN: FileCheck --input-file=%t.s %s

int foo() __attribute__((aligned(1024)));
int foo() { }

// CHECK:.align 10, 0x90
// CHECK:.globl __Z3foov
// CHECK:__Z3foov:


class C {
  virtual void bar1() __attribute__((aligned(2)));
  virtual void bar2() __attribute__((aligned(1024)));
} c;

void C::bar1() { }

// CHECK:.align 1, 0x90
// CHECK-NEXT:.globl __ZN1C4bar1Ev
// CHECK-NEXT:__ZN1C4bar1Ev:


void C::bar2() { }

// CHECK:.align  10, 0x90
// CHECK-NEXT:.globl __ZN1C4bar2Ev
// CHECK-NEXT:__ZN1C4bar2Ev:
