// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vptr -emit-llvm -o - %s | FileCheck %s

struct A {
  A();
  virtual void f();
};

struct B : virtual A {
  B();
};

struct C : virtual A {
  C();
};

namespace {

struct D : B, C {
  D();
  virtual void f();
};

}

A::A() {}
B::B() {}
C::C() {}
D::D() {}

void A::f() {
}

void D::f() {
}

// CHECK: define void @_Z2afP1A
void af(A *a) {
  // CHECK: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"1A")
  // CHECK-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ]*]]

  // CHECK: [[TRAPBB]]
  // CHECK-NEXT: call void @llvm.trap()
  // CHECK-NEXT: unreachable

  // CHECK: [[CONTBB]]
  // CHECK: call void %
  a->f();
}

// CHECK: define internal void @_Z2dfPN12_GLOBAL__N_11DE
void df(D *d) {
  // CHECK: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"[{{.*}}cfi-vptr.cpp]N12_GLOBAL__N_11DE")
  d->f();
}

D d;

void foo() {
  df(&d);
}

// CHECK-DAG: !{!"1A", [3 x i8*]* @_ZTV1A, i64 16}
// CHECK-DAG: !{!"1A", [5 x i8*]* @_ZTCN12_GLOBAL__N_11DE0_1B, i64 32}
// CHECK-DAG: !{!"1B", [5 x i8*]* @_ZTCN12_GLOBAL__N_11DE0_1B, i64 32}
// CHECK-DAG: !{!"1A", [9 x i8*]* @_ZTCN12_GLOBAL__N_11DE8_1C, i64 64}
// CHECK-DAG: !{!"1C", [9 x i8*]* @_ZTCN12_GLOBAL__N_11DE8_1C, i64 32}
// CHECK-DAG: !{!"1A", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// CHECK-DAG: !{!"1B", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// CHECK-DAG: !{!"1C", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 72}
// CHECK-DAG: !{!"[{{.*}}cfi-vptr.cpp]N12_GLOBAL__N_11DE", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// CHECK-DAG: !{!"1A", [5 x i8*]* @_ZTV1B, i64 32}
// CHECK-DAG: !{!"1B", [5 x i8*]* @_ZTV1B, i64 32}
// CHECK-DAG: !{!"1A", [5 x i8*]* @_ZTV1C, i64 32}
// CHECK-DAG: !{!"1C", [5 x i8*]* @_ZTV1C, i64 32}
