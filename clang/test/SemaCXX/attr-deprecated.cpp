// RUN: %clang_cc1 %s -verify -fsyntax-only
class A {
  void f() __attribute__((deprecated));
  void g(A* a);
  void h(A* a) __attribute__((deprecated));

  int b __attribute__((deprecated));
};

void A::g(A* a)
{
  f(); // expected-warning{{'f' is deprecated}}
  a->f(); // expected-warning{{'f' is deprecated}}
  
  (void)b; // expected-warning{{'b' is deprecated}}
  (void)a->b; // expected-warning{{'b' is deprecated}}
}

void A::h(A* a)
{
  f();
  a->f();
  
  (void)b;
  (void)a->b;
}

struct B {
  virtual void f() __attribute__((deprecated));
  void g();
};

void B::g() {
  f();
  B::f(); // expected-warning{{'f' is deprecated}}
}

struct C : B {
  virtual void f();
  void g();
};

void C::g() {
  f();
  C::f();
  B::f(); // expected-warning{{'f' is deprecated}}
}

void f(B* b, C *c) {
  b->f();
  b->B::f(); // expected-warning{{'f' is deprecated}}
  
  c->f();
  c->C::f();
  c->B::f(); // expected-warning{{'f' is deprecated}}
}

struct D {
  virtual void f() __attribute__((deprecated));
};

void D::f() { }

void f(D* d) {
  d->f();
}


// Overloaded namespace members.
namespace test1 {
  void foo(int) __attribute__((deprecated));
  void test1() { foo(10); } // expected-warning {{deprecated}}
  void foo(short) __attribute__((deprecated));
  void test2(short s) { foo(s); } // expected-warning {{deprecated}}
  void foo(long);
  void test3(long l) { foo(l); }
  struct A {
    friend void foo(A*) __attribute__((deprecated));
  };
  void test4(A *a) { foo(a); } // expected-warning {{deprecated}}

  namespace ns {
    struct Foo {};
    void foo(const Foo &f) __attribute__((deprecated));
  }
  void test5() {
    foo(ns::Foo()); // expected-warning {{deprecated}}
  }
}

// Overloaded class members.
namespace test2 {
  struct A {
    void foo(int) __attribute__((deprecated));
    void foo(long);
    static void bar(int) __attribute__((deprecated));
    static void bar(long);

    void test2(int i, long l);
  };
  void test1(int i, long l) {
    A a;
    a.foo(i); // expected-warning {{deprecated}}
    a.foo(l);
    a.bar(i); // expected-warning {{deprecated}}
    a.bar(l);
    A::bar(i); // expected-warning {{deprecated}}
    A::bar(l);
  }

  void A::test2(int i, long l) {
    foo(i); // expected-warning {{deprecated}}
    foo(l);
    bar(i); // expected-warning {{deprecated}}
    bar(l);
  }
}

// Overloaded operators.
namespace test3 {
  struct A {
    void operator*(const A &);
    void operator*(int) __attribute__((deprecated));
    void operator-(const A &) const;
  };
  void operator+(const A &, const A &);
  void operator+(const A &, int) __attribute__((deprecated));
  void operator-(const A &, int) __attribute__((deprecated));

  void test() {
    A a, b;
    a + b;
    a + 1; // expected-warning {{deprecated}}
    a - b;
    a - 1; // expected-warning {{deprecated}}
    a * b;
    a * 1; // expected-warning {{deprecated}}
  }
}

// Overloaded operator call.
namespace test4 {
  struct A {
    typedef void (*intfn)(int);
    typedef void (*unintfn)(unsigned);
    operator intfn() __attribute__((deprecated));
    operator unintfn();
    void operator ()(A &) __attribute__((deprecated));
    void operator ()(const A &);
  };

  void test() {
    A a;
    a(1); // expected-warning {{deprecated}}
    a(1U);

    A &b = a;
    const A &c = a;
    a(b); // expected-warning {{deprecated}}
    a(c);
  }
}

namespace test5 {
  struct A {
    operator int() __attribute__((deprecated));
    operator long();
  };
  void test1(A a) {
    int i = a; // expected-warning {{deprecated}}
    long l = a;
  }

  void foo(int);
  void foo(void*);
  void bar(long);
  void bar(void*);
  void test2(A a) {
    foo(a); // expected-warning {{deprecated}}
    bar(a);
  }

  struct B {
    int myInt;
    long myLong;

    B(A &a) :
      myInt(a), // expected-warning {{deprecated}}
      myLong(a)
    {}
  };
}
