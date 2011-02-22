// RUN: %clang_cc1 -fsyntax-only -verify %s

class X{
public:
  enum E {Enumerator}; // expected-note 2{{declared here}}
  int f();
  static int mem;
  static float g();
};

void test(X* xp, X x) {
  int i1 = x.f();
  int i2 = xp->f();
  x.E; // expected-error{{cannot refer to type member 'E' in 'X' with '.'}}
  xp->E; // expected-error{{cannot refer to type member 'E' in 'X' with '->'}}
  int i3 = x.Enumerator;
  int i4 = xp->Enumerator;
  x.mem = 1;
  xp->mem = 2;
  float f1 = x.g();
  float f2 = xp->g();
}

struct A {
 int f0;
};
struct B {
 A *f0();
};
int f0(B *b) {
  return b->f0->f0; // expected-error{{perhaps you meant to call it with no arguments}}
}

int i;

namespace C {
  int i;
}

void test2(X *xp) {
  xp->::i = 7; // expected-error{{qualified member access refers to a member in the global namespace}}
  xp->C::i = 7; // expected-error{{qualified member access refers to a member in namespace 'C'}}
}


namespace test3 {
  struct NamespaceDecl;

  struct NamedDecl {
    void *getIdentifier() const;
  };

  struct NamespaceDecl : NamedDecl {
    bool isAnonymousNamespace() const {
      return !getIdentifier();
    }
  };
}

namespace test4 {
  class X {
  protected:
    template<typename T> void f(T);
  };

  class Y : public X {
  public:
    using X::f;
  };

  void test_f(Y y) {
    y.f(17);
  }
}

namespace test5 {
  struct A {
    template <class T> void foo();
  };

  void test0(int x) {
    x.A::foo<int>(); // expected-error {{'int' is not a structure or union}}
  }

  void test1(A *x) {
    x.A::foo<int>(); // expected-error {{'test5::A *' is a pointer}}
  }

  void test2(A &x) {
    x->A::foo<int>(); // expected-error {{'test5::A' is not a pointer}}
  }
}

namespace PR7508 {
  struct A {
    struct CleanupScope {};
    void PopCleanupBlock();
  };

  void foo(A &a) {
    a.PopCleanupScope(); // expected-error{{no member named 'PopCleanupScope' in 'PR7508::A'}}
  }
}

namespace rdar8231724 {
  namespace N {
    template<typename T> struct X1;
    int i;
  }

  struct X { };
  struct Y : X { };

  void f(Y *y) {
    y->N::X1<int>; // expected-error{{'rdar8231724::N::X1' is not a member of class 'rdar8231724::Y'}}
  }
}

namespace PR9025 {
  struct S { int x; };
  S fun();
  int fun(int i);
  int g() {
    return fun.x; // expected-error{{base of member reference is an overloaded function; perhaps you meant to call it with no arguments?}}
  }

  S fun2(); // expected-note{{possibly valid overload here}}
  S fun2(int i); // expected-note{{possibly valid overload here}}
  int g2() {
    return fun2.x; // expected-error{{base of member reference is an overloaded function; perhaps you meant to call it?}}
  }

  S fun3(int i=0);
  int fun3(int i, int j);
  int g3() {
    return fun3.x; // expected-error{{base of member reference is an overloaded function; perhaps you meant to call it with no arguments?}}
  }
}
