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
  Status& operator=(const Status& x);
  inline void Update(const Status& new_status) {
    if (ok()) {
      *this = new_status; //no-warning
    }
  }
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

namespace PR17587 {
struct [[clang::warn_unused_result]] Status;

struct Foo {
  Status Bar();
};

struct Status {};

void Bar() {
  Foo f;
  f.Bar(); // expected-warning {{ignoring return value}}
};

}

namespace PR18571 {
// Unevaluated contexts should not trigger unused result warnings.
template <typename T>
auto foo(T) -> decltype(f(), bool()) { // Should not warn.
  return true;
}

void g() {
  foo(1);
}
}

namespace std {
class type_info { };
}

namespace {
// The typeid expression operand is evaluated only when the expression type is
// a glvalue of polymorphic class type.

struct B {
  virtual void f() {}
};

struct D : B {
  void f() override {}
};

struct C {};

void g() {
  // The typeid expression operand is evaluated only when the expression type is
  // a glvalue of polymorphic class type; otherwise the expression operand is not
  // evaluated and should not trigger a diagnostic.
  D d;
  C c;
  (void)typeid(f(), c); // Should not warn.
  (void)typeid(f(), d); // expected-warning {{ignoring return value}}

  // The sizeof expression operand is never evaluated.
  (void)sizeof(f(), c); // Should not warn.

   // The noexcept expression operand is never evaluated.
  (void)noexcept(f(), false); // Should not warn.
}
}
