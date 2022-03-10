// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCallGraph %s -fblocks -std=c++14 2>&1 | FileCheck %s

int get5() {
  return 5;
}

int add(int val1, int val2) {
  return val1 + val2;
}

int test_add() {
  return add(10, get5());
}

static void mmm(int y) {
  if (y != 0)
      y++;
  y = y/y;
}

static int foo(int x, int y) {
    mmm(y);
    if (x != 0)
      x++;
    return 5/x;
}

void aaa() {
  foo(1,2);
}

void bbb(int y) {
  int x = (y > 2);
  ^ {
      foo(x, y);
  }();
}
void ccc();
void ddd() { ccc(); }
void ccc() {}

void eee();
void eee() {}
void fff() { eee(); }

// This test case tests that forward declaration for the top-level function
// does not affect call graph construction.
void do_nothing() {}
void test_single_call();
void test_single_call() {
  do_nothing();
}

namespace SomeNS {
template<typename T>
void templ(T t) {
  ccc();
}

template<>
void templ<double>(double t) {
  eee();
}

void templUser() {
  templ(5);
  templ(5.5);
}
}

namespace Lambdas {
  void Callee(){}

  void f1() {
    [](int i) {
      Callee();
    }(1);
    [](auto i) {
      Callee();
    }(1);
  }
}

namespace CallDecl {
  void SomeDecl();
  void SomeOtherDecl();
  void SomeDef() {}

  void Caller() {
    SomeDecl();
    SomeOtherDecl();
  }

  void SomeOtherDecl() {
    SomeDef();
  }
}

// CHECK:--- Call graph Dump ---
// CHECK-NEXT: {{Function: < root > calls: get5 add test_add mmm foo aaa < > bbb ddd ccc eee fff do_nothing test_single_call SomeNS::templ SomeNS::templ SomeNS::templUser Lambdas::Callee Lambdas::f1 Lambdas::f1\(\)::\(anonymous class\)::operator\(\) Lambdas::f1\(\)::\(anonymous class\)::operator\(\) CallDecl::SomeDef CallDecl::Caller CallDecl::SomeDecl CallDecl::SomeOtherDecl $}}
// CHECK-NEXT: {{Function: CallDecl::Caller calls: CallDecl::SomeDecl CallDecl::SomeOtherDecl $}}
// CHECK-NEXT: {{Function: CallDecl::SomeOtherDecl calls: CallDecl::SomeDef $}}
// CHECK-NEXT: {{Function: CallDecl::SomeDecl calls: $}}
// CHECK-NEXT: {{Function: CallDecl::SomeDef calls: $}}
// CHECK-NEXT: {{Function: Lambdas::f1 calls: Lambdas::f1\(\)::\(anonymous class\)::operator\(\) Lambdas::f1\(\)::\(anonymous class\)::operator\(\) $}}
// CHECK-NEXT: {{Function: Lambdas::f1\(\)::\(anonymous class\)::operator\(\) calls: Lambdas::Callee $}}
// CHECK-NEXT: {{Function: Lambdas::f1\(\)::\(anonymous class\)::operator\(\) calls: Lambdas::Callee $}}
// CHECK-NEXT: {{Function: Lambdas::Callee calls: $}}
// CHECK-NEXT: {{Function: SomeNS::templUser calls: SomeNS::templ SomeNS::templ $}}
// CHECK-NEXT: {{Function: SomeNS::templ calls: eee $}}
// CHECK-NEXT: {{Function: SomeNS::templ calls: ccc $}}
// CHECK-NEXT: {{Function: test_single_call calls: do_nothing $}}
// CHECK-NEXT: {{Function: do_nothing calls: $}}
// CHECK-NEXT: {{Function: fff calls: eee $}}
// CHECK-NEXT: {{Function: eee calls: $}}
// CHECK-NEXT: {{Function: ddd calls: ccc $}}
// CHECK-NEXT: {{Function: ccc calls: $}}
// CHECK-NEXT: {{Function: bbb calls: < > $}}
// CHECK-NEXT: {{Function: < > calls: foo $}}
// CHECK-NEXT: {{Function: aaa calls: foo $}}
// CHECK-NEXT: {{Function: foo calls: mmm $}}
// CHECK-NEXT: {{Function: mmm calls: $}}
// CHECK-NEXT: {{Function: test_add calls: add get5 $}}
// CHECK-NEXT: {{Function: add calls: $}}
// CHECK-NEXT: {{Function: get5 calls: $}}
