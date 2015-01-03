// RUN: %clang_cc1 -fsyntax-only -verify %s

class c {
  virtual void f1(const char* a, ...)
    __attribute__ (( __format__(__printf__,2,3) )) = 0;
  virtual void f2(const char* a, ...)
    __attribute__ (( __format__(__printf__,2,3) )) {}
};

template <typename T> class X {
  template <typename S> void X<S>::f() __attribute__((locks_excluded())); // expected-error{{nested name specifier 'X<S>::' for declaration does not refer into a class, class template or class template partial specialization}} \
                                                                          // expected-warning{{attribute locks_excluded ignored, because it is not attached to a declaration}}
};

namespace PR17666 {
  const int A = 1;
  typedef int __attribute__((__aligned__(A))) T1;
  int check1[__alignof__(T1) == 1 ? 1 : -1];

  typedef int __attribute__((aligned(int(1)))) T1;
  typedef int __attribute__((aligned(int))) T2; // expected-error {{expected '(' for function-style cast}}
}

__attribute((typename)) int x; // expected-warning {{unknown attribute 'typename' ignored}}
