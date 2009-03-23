// RUN: clang -fsyntax-only -verify %s -std=c++0x

#ifndef __GXX_EXPERIMENTAL_CXX0X__
#define __CONCAT(__X, __Y) __CONCAT1(__X, __Y)
#define __CONCAT1(__X, __Y) __X ## __Y

#define static_assert(__b, __m) \
  typedef int __CONCAT(__sa, __LINE__)[__b ? 1 : -1]
#endif

class C {
    virtual void f() = 0; // expected-note {{pure virtual function 'f'}}
};

static_assert(__is_abstract(C), "C has a pure virtual function");

class D : C {
};

static_assert(__is_abstract(D), "D inherits from an abstract class");

class E : D {
    virtual void f();
};

static_assert(!__is_abstract(E), "E inherits from an abstract class but implements f");

C *d = new C; // expected-error {{allocation of an object of abstract type 'C'}}

C c; // expected-error {{variable type 'C' is an abstract class}}
void t1(C c); // expected-error {{parameter type 'C' is an abstract class}}
void t2(C); // expected-error {{parameter type 'C' is an abstract class}}

struct S {
  C c; // expected-error {{field type 'C' is an abstract class}}
};

