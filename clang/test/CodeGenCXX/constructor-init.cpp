// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -o - | FileCheck %s

extern "C" int printf(...);

struct M {
  M() { printf("M()\n"); }
  M(int i) { iM = i; printf("M(%d)\n", i); }
  int iM;
  void MPR() {printf("iM = %d\n", iM); };
};

struct P {
  P() { printf("P()\n"); }
  P(int i) { iP = i; printf("P(%d)\n", i); }
  int iP;
  void PPR() {printf("iP = %d\n", iP); };
};

struct Q {
  Q() { printf("Q()\n"); }
  Q(int i) { iQ = i; printf("Q(%d)\n", i); }
  int iQ;
  void QPR() {printf("iQ = %d\n", iQ); };
};

struct N : M , P, Q {
  N() : f1(1.314), P(2000), ld(00.1234+f1), M(1000), Q(3000),
        d1(3.4567), i1(1234), m1(100) { printf("N()\n"); }
  M m1;
  M m2;
  float f1;
  int i1;
  float d1;
  void PR() {
    printf("f1 = %f d1 = %f i1 = %d ld = %f \n", f1,d1,i1, ld); 
    MPR();
    PPR();
    QPR();
    printf("iQ = %d\n", iQ);
    printf("iP = %d\n", iP);
    printf("iM = %d\n", iM);
    // FIXME. We don't yet support this syntax.
    // printf("iQ = %d\n", (*this).iQ);
    printf("iQ = %d\n", this->iQ);
    printf("iP = %d\n", this->iP);
    printf("iM = %d\n", this->iM);
  }
  float ld;
  float ff;
  M arr_m[3];
  P arr_p[1][3];
  Q arr_q[2][3][4];
};

int main() {
  M m1;

  N n1;
  n1.PR();
}

// PR5826
template <class T> struct A {
  A() {}
  A(int) {}
  A(const A&) {}
  ~A() {}
  operator int() {return 0;}
};

// CHECK: define void @_Z1fv()
void f() {
  // CHECK: call void @_ZN1AIsEC1Ei
  A<short> a4 = 97;

  // CHECK-NEXT: store i32 17
  int i = 17;

  // CHECK-NEXT: call void @_ZN1AIsED1Ev
  // CHECK-NOT: call void @_ZN1AIsED1Ev
  // CHECK: ret void
}
