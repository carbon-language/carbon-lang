// RUN: clang-cc -triple x86_64-apple-darwin -std=c++0x -O0 -S %s -o %t-64.s &&
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s &&
// RUN: clang-cc -triple i386-apple-darwin -std=c++0x -O0 -S %s -o %t-32.s &&
// RUN: FileCheck -check-prefix LP32 -input-file=%t-32.s %s &&
// RUN: true

extern "C" int printf(...);

struct S {
  S() : iS(1), fS(1.23) {};
  ~S(){printf("S::~S(%d, %f)\n", iS, fS); };
  int iS;
  float fS;
};

struct Q {
  Q() : iQ(2), dQ(2.34) {};
  ~Q(){printf("Q::~Q(%d, %f)\n", iQ, dQ); };
  int iQ;
  double dQ;
};

struct P {
  P() : fP(3.45) , iP(3) {};
  ~P(){printf("P::~P(%d, %f)\n", iP, fP); };
  float fP;
  int iP;
};

struct M  : Q, P {
  S s;

  Q q;

  P p; 

};

M gm;

int main() {M m1;}

// CHECK-LP64:  call	__ZN1MC1Ev
// CHECK-LP64:  call	__ZN1MD1Ev
// CHECK-LP64:  .globl	__ZN1MD1Ev
// CHECK-LP64-NEXT:  .weak_definition __ZN1MD1Ev
// CHECK-LP64-NEXT:  __ZN1MD1Ev:


// CHECK-LP32:  call	L__ZN1MC1Ev
// CHECK-LP32:  call	L__ZN1MD1Ev
// CHECK-LP32:  .globl	__ZN1MD1Ev
// CHECK-LP32-NEXT:  .weak_definition __ZN1MD1Ev
// CHECK-LP32-NEXT:  __ZN1MD1Ev:
