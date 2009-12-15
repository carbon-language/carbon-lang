// RUN: %clang_cc1 -ast-dump %s 2>&1 | FileCheck %s

// CHECK: example0
void example0() {
  double d = 2.0;
  // CHECK: double &rd =
  // CHECK-NEXT: DeclRefExpr
  double &rd = d;
  // CHECK: double const &rcd =
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'double const' <NoOp>
  const double &rcd = d;
}

struct A { };
struct B : A { } b;

// CHECK: example1
void example1() {
  // CHECK: struct A &ra =
  // CHECK: ImplicitCastExpr{{.*}}'struct A' <DerivedToBase> lvalue
  A &ra = b;
  // CHECK: struct A const &rca =
  // CHECK: ImplicitCastExpr{{.*}}'struct A const' <NoOp>
  // CHECK: ImplicitCastExpr{{.*}}'struct A' <DerivedToBase>
  const A& rca = b;
}

extern B f();

struct X {
  operator B();
} x;

// CHECK: example2
void example2() {
  // CHECK: struct A const &rca =
  // CHECK: ImplicitCastExpr{{.*}}'struct A const' <NoOp>
  // CHECK: ImplicitCastExpr{{.*}}'struct A' <DerivedToBase>
  // CHECK: CallExpr{{.*}}struct B
  const A &rca = f(); 
  // CHECK: struct A const &r =
  // CHECK: ImplicitCastExpr{{.*}}'struct A const' <NoOp>
  // CHECK: ImplicitCastExpr{{.*}}'struct A' <DerivedToBase>
  // CHECK: CXXMemberCallExpr{{.*}}'struct B'
  const A& r = x;
}

// CHECK: example3
void example3() {
  // CHECK: double const &rcd2 =
  // CHECK: ImplicitCastExpr{{.*}}<IntegralToFloating>
  const double& rcd2 = 2; 
}
