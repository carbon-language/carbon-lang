// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s


// CHECK: _Z3fooRi(i32* inreg
void __attribute__ ((regparm (1)))  foo(int &a) {
}

struct S1 {
  int x;
  S1(const S1 &y);
};

void __attribute__((regparm(3))) foo2(S1 a, int b);
// CHECK: declare void @_Z4foo22S1i(%struct.S1* inreg, i32 inreg)
void bar2(S1 a, int b) {
  foo2(a, b);
}

struct S2 {
  int x;
};

void __attribute__((regparm(3))) foo3(struct S2 a, int b);
// declare void @_Z4foo12S1i(i32 inreg, i32 inreg) optsize
void bar3(struct S2 a, int b) {
  foo3(a, b);
}
