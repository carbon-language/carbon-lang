// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NDIAG %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=DIAG --check-prefix=DIAG-ABORT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-vcall -fsanitize-recover=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=DIAG --check-prefix=DIAG-RECOVER %s

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

// DIAG: @[[SRC:.*]] = private unnamed_addr constant [{{.*}} x i8] c"{{.*}}cfi-vcall.cpp\00", align 1
// DIAG: @[[TYPE:.*]] = private unnamed_addr constant { i16, i16, [4 x i8] } { i16 -1, i16 0, [4 x i8] c"'A'\00" }
// DIAG: @[[BADTYPESTATIC:.*]] = private unnamed_addr global { { [{{.*}} x i8]*, i32, i32 }, { i16, i16, [4 x i8] }*, i8 } { { [{{.*}} x i8]*, i32, i32 } { [{{.*}} x i8]* @[[SRC]], i32 58, i32 3 }, { i16, i16, [4 x i8] }* @[[TYPE]], i8 0 }

// CHECK: define void @_Z2afP1A
void af(A *a) {
  // CHECK: [[P:%[^ ]*]] = call i1 @llvm.bitset.test(i8* [[VT:%[^ ]*]], metadata !"1A")
  // CHECK-NEXT: br i1 [[P]], label %[[CONTBB:[^ ,]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK: [[TRAPBB]]
  // NDIAG-NEXT: call void @llvm.trap()
  // NDIAG-NEXT: unreachable
  // DIAG-NEXT: [[VTINT:%[^ ]*]] = ptrtoint i8* [[VT]] to i64
  // DIAG-ABORT-NEXT: call void @__ubsan_handle_cfi_bad_type_abort(i8* bitcast ({{.*}} @[[BADTYPESTATIC]] to i8*), i64 [[VTINT]])
  // DIAG-ABORT-NEXT: unreachable
  // DIAG-RECOVER-NEXT: call void @__ubsan_handle_cfi_bad_type(i8* bitcast ({{.*}} @[[BADTYPESTATIC]] to i8*), i64 [[VTINT]])
  // DIAG-RECOVER-NEXT: br label %[[CONTBB]]

  // CHECK: [[CONTBB]]
  // CHECK: call void %
  a->f();
}

// CHECK: define internal void @_Z3df1PN12_GLOBAL__N_11DE
void df1(D *d) {
  // CHECK: {{%[^ ]*}} = call i1 @llvm.bitset.test(i8* {{%[^ ]*}}, metadata !"[{{.*}}cfi-vcall.cpp]N12_GLOBAL__N_11DE")
  d->f();
}

// CHECK: define internal void @_Z3df2PN12_GLOBAL__N_11DE
__attribute__((no_sanitize("cfi")))
void df2(D *d) {
  // CHECK-NOT: call i1 @llvm.bitset.test
  d->f();
}

// CHECK: define internal void @_Z3df3PN12_GLOBAL__N_11DE
__attribute__((no_sanitize("address"))) __attribute__((no_sanitize("cfi-vcall")))
void df3(D *d) {
  // CHECK-NOT: call i1 @llvm.bitset.test
  d->f();
}

D d;

void foo() {
  df1(&d);
  df2(&d);
  df3(&d);
}

// CHECK-DAG: !{!"1A", [3 x i8*]* @_ZTV1A, i64 16}
// CHECK-DAG: !{!"1A", [5 x i8*]* @_ZTCN12_GLOBAL__N_11DE0_1B, i64 32}
// CHECK-DAG: !{!"1B", [5 x i8*]* @_ZTCN12_GLOBAL__N_11DE0_1B, i64 32}
// CHECK-DAG: !{!"1A", [9 x i8*]* @_ZTCN12_GLOBAL__N_11DE8_1C, i64 64}
// CHECK-DAG: !{!"1C", [9 x i8*]* @_ZTCN12_GLOBAL__N_11DE8_1C, i64 32}
// CHECK-DAG: !{!"1A", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// CHECK-DAG: !{!"1B", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// CHECK-DAG: !{!"1C", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 72}
// CHECK-DAG: !{!"[{{.*}}cfi-vcall.cpp]N12_GLOBAL__N_11DE", [10 x i8*]* @_ZTVN12_GLOBAL__N_11DE, i64 32}
// CHECK-DAG: !{!"1A", [5 x i8*]* @_ZTV1B, i64 32}
// CHECK-DAG: !{!"1B", [5 x i8*]* @_ZTV1B, i64 32}
// CHECK-DAG: !{!"1A", [5 x i8*]* @_ZTV1C, i64 32}
// CHECK-DAG: !{!"1C", [5 x i8*]* @_ZTV1C, i64 32}
