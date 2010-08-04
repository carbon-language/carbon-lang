// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef typeof(sizeof(int)) size_t;

// PR7803
namespace test0 {
  class A {
  public:
    static void operator delete(void *p) {};
    virtual ~A();
  };

  class B : protected A {
  public:
    ~B();
  };

  class C : protected B {
  public:
    using B::operator delete;
    ~C();
  };

  // Shouldn't have an error.
  C::~C() {}
}

namespace test1 {
  class A {
  public:
    static void operator delete(void *p) {}; // expected-note {{member 'operator delete' declared here}}
    virtual ~A();
  };

  class B : protected A {
  public:
    static void operator delete(void *, size_t) {}; // expected-note {{member 'operator delete' declared here}}
    ~B();
  };

  class C : protected B {
  public:
    using A::operator delete;
    using B::operator delete;

    ~C(); // expected-error {{multiple suitable 'operator delete' functions in 'C'}}
  };
}
