// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

int f() __attribute__((warn_unused_result));

struct S {
  void t() const;
};
S g1() __attribute__((warn_unused_result));
S *g2() __attribute__((warn_unused_result));
S &g3() __attribute__((warn_unused_result));

void test() {
  f(); // expected-warning {{ignoring return value}}
  g1(); // expected-warning {{ignoring return value}}
  g2(); // expected-warning {{ignoring return value}}
  g3(); // expected-warning {{ignoring return value}}

  (void)f();
  (void)g1();
  (void)g2();
  (void)g3();

  if (f() == 0) return;

  g1().t();
  g2()->t();
  g3().t();

  int i = f();
  S s1 = g1();
  S *s2 = g2();
  S &s3 = g3();
  const S &s4 = g1();
}

struct X {
 int foo() __attribute__((warn_unused_result));
};

void bah() {
  X x, *x2;
  x.foo(); // expected-warning {{ignoring return value}}
  x2->foo(); // expected-warning {{ignoring return value}}
}

namespace warn_unused_CXX11 {
struct [[clang::warn_unused_result]] Status {
  bool ok() const;
};
Status DoSomething();
Status& DoSomethingElse();
Status* DoAnotherThing();
Status** DoYetAnotherThing();
void lazy() {
  Status s = DoSomething();
  if (!s.ok()) return;
  Status &rs = DoSomethingElse();
  if (!rs.ok()) return;
  Status *ps = DoAnotherThing();
  if (!ps->ok()) return;
  Status **pps = DoYetAnotherThing();
  if (!(*pps)->ok()) return;

  (void)DoSomething();
  (void)DoSomethingElse();
  (void)DoAnotherThing();
  (void)DoYetAnotherThing();

  DoSomething(); // expected-warning {{ignoring return value}}
  DoSomethingElse(); // expected-warning {{ignoring return value}}
  DoAnotherThing(); // expected-warning {{ignoring return value}}
  DoYetAnotherThing();
}
}
