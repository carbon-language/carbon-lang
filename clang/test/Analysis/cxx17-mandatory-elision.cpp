// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

void clang_analyzer_eval(bool);

namespace variable_functional_cast_crash {

struct A {
  A(int) {}
};

void foo() {
  A a = A(0);
}

struct B {
  A a;
  B(): a(A(0)) {}
};

} // namespace variable_functional_cast_crash


namespace ctor_initializer {

struct S {
  int x, y, z;
};

struct T {
  S s;
  int w;
  T(int w): s(), w(w) {}
};

class C {
  T t;
public:
  C() : t(T(4)) {
    S s = {1, 2, 3};
    t.s = s;
    // FIXME: Should be TRUE in C++11 as well.
    clang_analyzer_eval(t.w == 4);
#if __cplusplus >= 201703L
    // expected-warning@-2{{TRUE}}
#else
    // expected-warning@-4{{UNKNOWN}}
#endif
  }
};

} // namespace ctor_initializer


namespace address_vector_tests {

template <typename T> struct AddressVector {
  T *buf[10];
  int len;

  AddressVector() : len(0) {}

  void push(T *t) {
    buf[len] = t;
    ++len;
  }
};

class ClassWithoutDestructor {
  AddressVector<ClassWithoutDestructor> &v;

public:
  ClassWithoutDestructor(AddressVector<ClassWithoutDestructor> &v) : v(v) {
    v.push(this);
  }

  ClassWithoutDestructor(ClassWithoutDestructor &&c) : v(c.v) { v.push(this); }
  ClassWithoutDestructor(const ClassWithoutDestructor &c) : v(c.v) {
    v.push(this);
  }
};

ClassWithoutDestructor make1(AddressVector<ClassWithoutDestructor> &v) {
  return ClassWithoutDestructor(v);
}
ClassWithoutDestructor make2(AddressVector<ClassWithoutDestructor> &v) {
  return make1(v);
}
ClassWithoutDestructor make3(AddressVector<ClassWithoutDestructor> &v) {
  return make2(v);
}

void testMultipleReturns() {
  AddressVector<ClassWithoutDestructor> v;
  ClassWithoutDestructor c = make3(v);

#if __cplusplus >= 201703L
  // FIXME: Both should be TRUE.
  clang_analyzer_eval(v.len == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == &c); // expected-warning{{FALSE}}
#else
  clang_analyzer_eval(v.len == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] != v.buf[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] != v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[2] != v.buf[3]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[3] != v.buf[4]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[4] == &c); // expected-warning{{TRUE}}
#endif
}

} // namespace address_vector_tests
