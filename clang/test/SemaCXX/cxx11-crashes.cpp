// RUN: %clang_cc1 -std=c++11 -verify %s

// rdar://12240916 stack overflow.
namespace rdar12240916 {

struct S2 {
  S2(const S2&);
  S2();
};

struct S { // expected-note {{not complete}}
  S x; // expected-error {{incomplete type}}
  S2 y;
};

S foo() {
  S s;
  return s;
}

struct S3; // expected-note {{forward declaration}}

struct S4 {
  S3 x; // expected-error {{incomplete type}}
  S2 y;
};

struct S3 {
  S4 x;
  S2 y;
};

S4 foo2() {
  S4 s;
  return s;
}

}
