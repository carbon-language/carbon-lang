// RUN: %clang_cc1 -fsyntax-only -verify %s 
class X {
public:
  explicit X(const X&);
  X(int*); // expected-note 2{{candidate constructor}}
  explicit X(float*);
};

class Y : public X { };

void f(Y y, int *ip, float *fp) {
  X x1 = y; // expected-error{{no matching constructor for initialization of 'X'}}
  X x2 = 0;
  X x3 = ip;
  X x4 = fp; // expected-error{{no viable conversion}}
}

struct foo {
 void bar();
};

// PR3600
void test(const foo *P) { P->bar(); } // expected-error{{cannot initialize object parameter of type 'foo' with an expression of type 'foo const'}}

namespace PR6757 {
  struct Foo {
    Foo();
    Foo(Foo&); // expected-note{{candidate constructor not viable}}
  };

  struct Bar {
    operator const Foo&() const;
  };

  void f(Foo);

  void g(Foo foo) {
    f(Bar()); // expected-error{{no viable constructor copying parameter of type 'PR6757::Foo const'}}
    f(foo);
  }
}
