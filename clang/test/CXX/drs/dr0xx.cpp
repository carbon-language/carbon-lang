// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -pedantic-errors -Wno-bind-to-temporary-copy
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -pedantic-errors

namespace dr1 { // dr1: no
  namespace X { extern "C" void dr1_f(int a = 1); } // expected-note 2{{candidate}} expected-note {{conflicting}}
  namespace Y { extern "C" void dr1_f(int a = 2); } // expected-note 2{{candidate}} expected-note {{target}}
  using X::dr1_f; using Y::dr1_f;
  void g() {
    // FIXME: The first of these two should be accepted.
    dr1_f(0); // expected-error {{ambiguous}}
    dr1_f(); // expected-error {{ambiguous}}
  }
  namespace X {
    using Y::dr1_f; // expected-error {{conflicts with declaration already in scope}}
    void h() {
      // FIXME: The second of these two should be rejected.
      dr1_f(0);
      dr1_f();
    }
  }

  namespace X {
    void z(int);
  }
  void X::z(int = 1) {} // expected-note {{previous}}
  namespace X {
    void z(int = 2); // expected-error {{redefinition of default argument}}
  }
}

namespace dr3 { // dr3: yes
  template<typename T> struct A {};
  template<typename T> void f(T) { A<T> a; } // expected-note {{implicit instantiation}}
  template void f(int);
  template<> struct A<int> {}; // expected-error {{explicit specialization of 'dr3::A<int>' after instantiation}}
}

namespace dr4 { // dr4: yes
  extern "C" {
    static void dr4_f(int) {}
    static void dr4_f(float) {}
    void dr4_g(int) {} // expected-note {{previous}}
    void dr4_g(float) {} // expected-error {{conflicting types}}
  }
}

namespace dr5 { // dr5: yes
  struct A {} a;
  struct B {
    B(const A&);
    B(const B&);
  };
  const volatile B b = a;

  struct C { C(C&); };
  struct D : C {};
  struct E { operator D&(); } e;
  const C c = e;
}

namespace dr7 { // dr7: no
  class A { public: ~A(); };
  class B : virtual private A {};
  class C : public B {} c; // FIXME: should be rejected, ~A is inaccessible

  class X { ~X(); }; // expected-note {{here}}
  class Y : X { ~Y() {} }; // expected-error {{private destructor}}
}

namespace dr8 { // dr8: dup 45
  class A {
    struct U;
    static const int k = 5;
    void f();
    template<typename, int, void (A::*)()> struct T;

    T<U, k, &A::f> *g();
  };
  A::T<A::U, A::k, &A::f> *A::g() { return 0; }
}

namespace dr9 { // dr9: yes
  struct B {
  protected:
    int m; // expected-note {{here}}
    friend int R1();
  };
  struct N : protected B { // expected-note 2{{protected}}
    friend int R2();
  } n;
  int R1() { return n.m; } // expected-error {{protected base class}} expected-error {{protected member}}
  int R2() { return n.m; }
}

namespace dr10 { // dr10: dup 45
  class A {
    struct B {
      A::B *p;
    };
  };
}

namespace dr11 { // dr11: yes
  template<typename T> struct A : T {
    using typename T::U;
    U u;
  };
  template<typename T> struct B : T {
    using T::V;
    V v; // expected-error {{unknown type name}}
  };
  struct X { typedef int U; };
  A<X> ax;
}

namespace dr12 { // dr12: sup 239
  enum E { e };
  E &f(E, E = e);
  void g() {
    int &f(int, E = e);
    // Under DR12, these call two different functions.
    // Under DR239, they call the same function.
    int &b = f(e);
    int &c = f(1);
  }
}

namespace dr14 { // dr14: no
  namespace X { extern "C" int dr14_f(); } // expected-note {{candidate}}
  namespace Y { extern "C" int dr14_f(); } // expected-note {{candidate}}
  using namespace X;
  using namespace Y;
  // FIXME: This should be accepted, name lookup only finds one function (in two
  // different namespaces).
  int k = dr14_f(); // expected-error {{ambiguous}}

  class C {
    int k; // expected-note {{here}}
    friend int Y::dr14_f();
  } c;
  namespace Z {
    // FIXME: This should be accepted, this function is a friend.
    extern "C" int dr14_f() { return c.k; } // expected-error {{private}}
  }

  namespace X { typedef int T; typedef int U; } // expected-note {{candidate}}
  namespace Y { typedef int T; typedef long U; } // expected-note {{candidate}}
  T t; // ok, same type both times
  U u; // expected-error {{ambiguous}}
}

namespace dr15 { // dr15: yes
  template<typename T> void f(int); // expected-note {{previous}}
  template<typename T> void f(int = 0); // expected-error {{default arguments cannot be added}}
}

namespace dr16 { // dr16: yes
  class A { // expected-note {{here}}
    void f(); // expected-note {{here}}
    friend class C;
  };
  class B : A {}; // expected-note 4{{here}}
  class C : B {
    void g() {
      f(); // expected-error {{private member}} expected-error {{private base}}
      A::f(); // expected-error {{private member}} expected-error {{private base}}
    }
  };
}

namespace dr17 { // dr17: yes
  class A {
    int n;
    int f();
    struct C;
  };
  struct B : A {} b;
  int A::f() { return b.n; }
  struct A::C : A {
    int g() { return n; }
  };
}

namespace dr18 { // dr18: yes
  typedef void Void;
  void f(Void); // expected-error {{empty parameter list defined with a typedef of 'void'}}
}

namespace dr19 { // dr19: yes
  struct A {
    int n; // expected-note {{here}}
  };
  struct B : protected A { // expected-note {{here}}
  };
  struct C : B {} c;
  struct D : B {
    int get1() { return c.n; } // expected-error {{protected member}}
    int get2() { return ((A&)c).n; } // ok, A is an accessible base of B from here
  };
}

namespace dr20 { // dr20: yes
  class X {
  public:
    X();
  private:
    X(const X&); // expected-note {{here}}
  };
  X f();
  X x = f(); // expected-error {{private}}
}

namespace dr21 { // dr21: no
  template<typename T> struct A;
  struct X {
    // FIXME: We should reject these, per [temp.param]p9.
    template<typename T = int> friend struct A;
    template<typename T = int> friend struct B;
  };
}

namespace dr22 { // dr22: sup 481
  template<typename dr22_T = dr22_T> struct X; // expected-error {{unknown type name 'dr22_T'}}
  typedef int T;
  template<typename T = T> struct Y;
}

namespace dr23 { // dr23: yes
  template<typename T> void f(T, T); // expected-note {{candidate}}
  template<typename T> void f(T, int); // expected-note {{candidate}}
  void g() { f(0, 0); } // expected-error {{ambiguous}}
}

// dr24: na

namespace dr25 { // dr25: no
  struct A {
    void f() throw(int);
  };
  // FIXME: The initializations of g and i should be rejected.
  void (A::*f)() throw (int);
  void (A::*g)() throw () = f;
  void (A::*h)() throw (int, char) = f;
  void (A::*i)() throw () = &A::f;
  void (A::*j)() throw (int, char) = &A::f;
  void x() {
    // FIXME: The assignments to g and i should be rejected.
    g = f;
    h = f;
    i = &A::f;
    j = &A::f;
  }
}

namespace dr26 { // dr26: yes
  struct A { A(A, const A & = A()); }; // expected-error {{must pass its first argument by reference}}
  struct B {
    B(); // expected-note {{candidate}}
    B(const B &, B = B()); // expected-error {{no matching constructor}} expected-note {{candidate}} expected-note {{here}}
  };
}

namespace dr27 { // dr27: yes
  enum E { e } n;
  E &m = true ? n : n;
}

// dr28: na

namespace dr29 { // dr29: no
  void dr29_f0(); // expected-note {{here}}
  void g0() { void dr29_f0(); }
  extern "C++" void g0_cxx() { void dr29_f0(); }
  extern "C" void g0_c() { void dr29_f0(); } // expected-error {{different language linkage}}

  extern "C" void dr29_f1(); // expected-note {{here}}
  void g1() { void dr29_f1(); }
  extern "C" void g1_c() { void dr29_f1(); }
  extern "C++" void g1_cxx() { void dr29_f1(); } // expected-error {{different language linkage}}

  // FIXME: We should reject this.
  void g2() { void dr29_f2(); }
  extern "C" void dr29_f2();

  // FIXME: We should reject this.
  extern "C" void g3() { void dr29_f3(); }
  extern "C++" void dr29_f3();

  // FIXME: We should reject this.
  extern "C++" void g4() { void dr29_f4(); }
  extern "C" void dr29_f4();

  extern "C" void g5();
  extern "C++" void dr29_f5();
  void g5() {
    void dr29_f5(); // ok, g5 is extern "C" but we're not inside the linkage-specification here.
  }

  extern "C++" void g6();
  extern "C" void dr29_f6();
  void g6() {
    void dr29_f6(); // ok, g6 is extern "C" but we're not inside the linkage-specification here.
  }

  extern "C" void g7();
  extern "C++" void dr29_f7(); // expected-note {{here}}
  extern "C" void g7() {
    void dr29_f7(); // expected-error {{different language linkage}}
  }

  extern "C++" void g8();
  extern "C" void dr29_f8(); // expected-note {{here}}
  extern "C++" void g8() {
    void dr29_f8(); // expected-error {{different language linkage}}
  }
}

namespace dr30 { // dr30: sup 468
  struct A {
    template<int> static int f();
  } a, *p = &a;
  int x = A::template f<0>();
  int y = a.template f<0>();
  int z = p->template f<0>();
#if __cplusplus < 201103L
  // FIXME: It's not clear whether DR468 applies to C++98 too.
  // expected-error@-5 {{'template' keyword outside of a template}}
  // expected-error@-5 {{'template' keyword outside of a template}}
  // expected-error@-5 {{'template' keyword outside of a template}}
#endif
}

namespace dr31 { // dr31: yes
  class X {
  private:
    void operator delete(void*); // expected-note {{here}}
  };
  // We would call X::operator delete if X() threw (even though it can't,
  // and even though we allocated the X using ::operator delete).
  X *p = new X; // expected-error {{private}}
}

// dr32: na

namespace dr33 { // dr33: yes
  namespace X { struct S; void f(void (*)(S)); } // expected-note {{candidate}}
  namespace Y { struct T; void f(void (*)(T)); } // expected-note {{candidate}}
  void g(X::S);
  template<typename Z> Z g(Y::T);
  void h() { f(&g); } // expected-error {{ambiguous}}
}

// dr34: na
// dr35: dup 178
// dr37: sup 475

namespace dr38 { // dr38: yes
  template<typename T> struct X {};
  template<typename T> X<T> operator+(X<T> a, X<T> b) { return a; }
  template X<int> operator+<int>(X<int>, X<int>);
}

namespace dr39 { // dr39: no
  namespace example1 {
    struct A { int &f(int); };
    struct B : A {
      using A::f;
      float &f(float);
    } b;
    int &r = b.f(0);
  }

  namespace example2 {
    struct A {
      int &x(int); // expected-note {{found}}
      static int &y(int); // expected-note {{found}}
    };
    struct V {
      int &z(int);
    };
    struct B : A, virtual V {
      using A::x; // expected-note {{found}}
      float &x(float);
      using A::y; // expected-note {{found}}
      static float &y(float);
      using V::z;
      float &z(float);
    };
    struct C : A, B, virtual V {} c;
    int &x = c.x(0); // expected-error {{found in multiple base classes}}
    // FIXME: This is valid, because we find the same static data member either way.
    int &y = c.y(0); // expected-error {{found in multiple base classes}}
    int &z = c.z(0);
  }

  namespace example3 {
    struct A { static int f(); };
    struct B : virtual A { using A::f; };
    struct C : virtual A { using A::f; };
    struct D : B, C {} d;
    int k = d.f();
  }

  namespace example4 {
    struct A { int n; }; // expected-note {{found}}
    struct B : A {};
    struct C : A {};
    struct D : B, C { int f() { return n; } }; // expected-error {{found in multiple base-class}}
  }
}

// dr40: na

namespace dr41 { // dr41: yes
  struct S f(S);
}

namespace dr42 { // dr42: yes
  struct A { static const int k = 0; };
  struct B : A { static const int k = A::k; };
}

// dr43: na

namespace dr44 { // dr44: yes
  struct A {
    template<int> void f();
    template<> void f<0>(); // expected-error {{explicit specialization of 'f' in class scope}}
  };
}

namespace dr45 { // dr45: yes
  class A {
    class B {};
    class C : B {};
    C c;
  };
}

namespace dr46 { // dr46: yes
  template<typename> struct A { template<typename> struct B {}; };
  template template struct A<int>::B<int>; // expected-error {{expected unqualified-id}}
}

namespace dr47 { // dr47: no
  template<typename T> struct A {
    friend void f() { T t; }
  };
  A<int> a;
  A<float> b;
#if __cplusplus < 201103L
  // expected-error@-5 {{redefinition}} expected-note@-5 {{previous}}
  // expected-note@-3 {{instantiation of}}
#else
  void f();
  // FIXME: We should produce some kind of error here. C++11 [temp.friend]p4
  // says we instantiate 'f' when it's odr-used, but that doesn't imply that
  // this is valid; we still have multiple definitions of 'f' even if we never
  // instantiate any of them.
  void g() { f(); }
#endif
}

namespace dr48 { // dr48: yes
  namespace {
    struct S {
      static const int m = 0;
      static const int n = 0;
      static const int o = 0;
    };
  }
  int a = S::m;
  // FIXME: We should produce a 'has internal linkage but is not defined'
  // diagnostic for 'S::n'.
  const int &b = S::n;
  const int S::o;
  const int &c = S::o;
}

namespace dr49 { // dr49: yes
  template<int*> struct A {}; // expected-note {{here}}
  int k;
#if __has_feature(cxx_constexpr)
  constexpr
#endif
  int *const p = &k;
  A<&k> a;
  A<p> b; // expected-error {{must have its address taken}}
#if __cplusplus < 201103L
  // expected-error@-2 {{internal linkage}}
  // expected-note@-5 {{here}}
#endif
}

namespace dr50 { // dr50: yes
  struct X; // expected-note {{forward}}
  extern X *p;
  X *q = (X*)p;
  X *r = static_cast<X*>(p);
  X *s = const_cast<X*>(p);
  X *t = reinterpret_cast<X*>(p);
  X *u = dynamic_cast<X*>(p); // expected-error {{incomplete}}
}
