// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

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

void t3(const C&);

void f() {
  C(); // expected-error {{allocation of an object of abstract type 'C'}}
  t3(C()); // expected-error {{allocation of an object of abstract type 'C'}}
}

C e1[2]; // expected-error {{variable type 'C' is an abstract class}}
C (*e2)[2]; // expected-error {{variable type 'C' is an abstract class}}
C (**e3)[2]; // expected-error {{variable type 'C' is an abstract class}}

void t4(C c[2]); // expected-error {{parameter type 'C' is an abstract class}}

void t5(void (*)(C)); // expected-error {{parameter type 'C' is an abstract class}}

typedef void (*Func)(C); // expected-error {{parameter type 'C' is an abstract class}}
void t6(Func);

class F {
  F a() { while (1) {} } // expected-error {{return type 'F' is an abstract class}}
    
  class D {
    void f(F c); // expected-error {{parameter type 'F' is an abstract class}}
  };

  union U {
    void u(F c); // expected-error {{parameter type 'F' is an abstract class}}
  };
    
  virtual void f() = 0; // expected-note {{pure virtual function 'f'}}
};

class Abstract;

void t7(Abstract a); // expected-error {{parameter type 'Abstract' is an abstract class}}

void t8() {
  void h(Abstract a); // expected-error {{parameter type 'Abstract' is an abstract class}}
}

namespace N {
void h(Abstract a); // expected-error {{parameter type 'Abstract' is an abstract class}}
}

class Abstract {
  virtual void f() = 0; // expected-note {{pure virtual function 'f'}}
};

// <rdar://problem/6854087>
class foo {
public:
  virtual foo *getFoo() = 0;
};

class bar : public foo {
public:
  virtual bar *getFoo();
};

bar x;

// <rdar://problem/6902298>
class A {
public:
  virtual void release() = 0;
  virtual void release(int count) = 0;
  virtual void retain() = 0;
};

class B : public A {
public:
  virtual void release();
  virtual void release(int count);
  virtual void retain();
};

void foo(void) {
  B b;
}

struct K {
 int f;
 virtual ~K();
};

struct L : public K {
 void f();
};
