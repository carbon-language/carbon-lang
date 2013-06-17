// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr100 { // dr100: yes
  template<const char *> struct A {}; // expected-note {{declared here}}
  template<const char (&)[4]> struct B {}; // expected-note {{declared here}}
  A<"foo"> a; // expected-error {{does not refer to any declaration}}
  B<"bar"> b; // expected-error {{does not refer to any declaration}}
}

namespace dr101 { // dr101: no
  // FIXME: This is valid.
  extern "C" void dr101_f(); // expected-note {{conflicting declaration}}
  typedef unsigned size_t;
  namespace X {
    extern "C" void dr101_f(); // expected-note {{target of using declaration}}
    typedef unsigned size_t;
  }
  using X::dr101_f; // expected-error {{conflicts with declaration already in scope}}
  using X::size_t;
}

namespace dr102 { // dr102: yes
  namespace A {
    template<typename T> T f(T a, T b) { return a + b; } // expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
  }
  namespace B {
    struct S {};
  }
  B::S operator+(B::S, B::S); // expected-note {{should be declared prior to the call site or in namespace 'dr102::B'}}
  template B::S A::f(B::S, B::S); // expected-note {{in instantiation of}}
}

// dr103: na
// dr104 FIXME: add codegen test
// dr105: na

namespace dr106 { // dr106: sup 540
  typedef int &r1;
  typedef r1 &r1;
  typedef const r1 r1;
  typedef const r1 &r1;

  typedef const int &r2;
  typedef r2 &r2;
  typedef const r2 r2;
  typedef const r2 &r2;
}

namespace dr107 { // dr107: yes
  struct S {};
  extern "C" S operator+(S, S) { return S(); }
}

namespace dr108 { // dr108: yes
  template<typename T> struct A {
    struct B { typedef int X; };
    B::X x; // expected-error {{missing 'typename'}}
    struct C : B { X x; }; // expected-error {{unknown type name}}
  };
  template<> struct A<int>::B { int X; };
}

namespace dr109 { // dr109: yes
  struct A { template<typename T> void f(T); };
  template<typename T> struct B : T {
    using T::template f; // expected-error {{using declaration can not refer to a template}}
    void g() { this->f<int>(123); } // expected-error {{use 'template'}}
  };
}

namespace dr111 { // dr111: dup 535
  struct A { A(); A(volatile A&, int = 0); A(A&, const char * = "foo"); };
  struct B : A { B(); }; // expected-note {{would lose const qualifier}} expected-note {{requires 0 arguments}}
  const B b1;
  B b2(b1); // expected-error {{no matching constructor}}
}

namespace dr112 { // dr112: yes
  struct T { int n; };
  typedef T Arr[1];

  const T a1[1] = {};
  volatile T a2[1] = {};
  const Arr a3 = {};
  volatile Arr a4 = {};
  template<const volatile T*> struct X {};
  X<a1> x1;
  X<a2> x2;
  X<a3> x3;
  X<a4> x4;
#if __cplusplus < 201103L
  // expected-error@-5 {{internal linkage}} expected-note@-10 {{here}}
  // expected-error@-4 {{internal linkage}} expected-note@-9 {{here}}
#else
  // FIXME: Test this somehow.
#endif
}

namespace dr113 { // dr113: yes
  extern void (*p)();
  void f() {
    no_such_function(); // expected-error {{undeclared}}
    p();
  }
  void g();
  void (*p)() = &g;
}

namespace dr114 { // dr114: yes
  struct A {
    virtual void f(int) = 0; // expected-note {{unimplemented}}
  };
  struct B : A {
    template<typename T> void f(T);
    void g() { f(0); }
  } b; // expected-error {{abstract}}
}

namespace dr115 { // dr115: yes
  template<typename T> int f(T); // expected-note +{{}}
  template<typename T> int g(T); // expected-note +{{}}
  template<typename T> int g(T, int); // expected-note +{{}}

  int k1 = f(&f); // expected-error {{no match}}
  int k2 = f(&f<int>);
  int k3 = f(&g<int>); // expected-error {{no match}}

  void h() {
    (void)&f; // expected-error {{address of overloaded function 'f' cannot be cast to type 'void'}}
    (void)&f<int>;
    (void)&g<int>; // expected-error {{address of overloaded function 'g' cannot be cast to type 'void'}}

    &f; // expected-error {{reference to overloaded function could not be resolved}}
    &f<int>; // expected-warning {{unused}}
    &g<int>; // expected-error {{reference to overloaded function could not be resolved}}
  }

  struct S {
    template<typename T> static int f(T);
    template<typename T> static int g(T);
    template<typename T> static int g(T, int);
  } s;

  int k4 = f(&s.f); // expected-error {{non-constant pointer to member}}
  int k5 = f(&s.f<int>);
  int k6 = f(&s.g<int>); // expected-error {{non-constant pointer to member}}

  void i() {
    (void)&s.f; // expected-error {{non-constant pointer to member}}
    (void)&s.f<int>;
    (void)&s.g<int>; // expected-error {{non-constant pointer to member}}

    &s.f; // expected-error {{non-constant pointer to member}}
    &s.f<int>; // expected-warning {{unused}}
    &s.g<int>; // expected-error {{non-constant pointer to member}}
  }

  struct T {
    template<typename T> int f(T);
    template<typename T> int g(T);
    template<typename T> int g(T, int);
  } t;

  int k7 = f(&s.f); // expected-error {{non-constant pointer to member}}
  int k8 = f(&s.f<int>);
  int k9 = f(&s.g<int>); // expected-error {{non-constant pointer to member}}

  void j() {
    (void)&s.f; // expected-error {{non-constant pointer to member}}
    (void)&s.f<int>;
    (void)&s.g<int>; // expected-error {{non-constant pointer to member}}

    &s.f; // expected-error {{non-constant pointer to member}}
    &s.f<int>; // expected-warning {{unused}}
    &s.g<int>; // expected-error {{non-constant pointer to member}}
  }

#if __cplusplus >= 201103L
  // Special case kicks in only if a template argument list is specified.
  template<typename T=int> void with_default(); // expected-note +{{}}
  int k10 = f(&with_default); // expected-error {{no matching function}}
  int k11 = f(&with_default<>);
  void k() {
    (void)&with_default; // expected-error {{overloaded function}}
    (void)&with_default<>;
    &with_default; // expected-error {{overloaded function}}
    &with_default<>; // expected-warning {{unused}}
  }
#endif
}

namespace dr116 { // dr116: yes
  template<int> struct A {};
  template<int N> void f(A<N>) {} // expected-note {{previous}}
  template<int M> void f(A<M>) {} // expected-error {{redefinition}}
  template<typename T> void f(A<sizeof(T)>) {} // expected-note {{previous}}
  template<typename U> void f(A<sizeof(U)>) {} // expected-error {{redefinition}}
}

// dr117: na
// dr118 FIXME: add codegen test
// dr119: na
// dr120: na

namespace dr121 { // dr121: yes
  struct X {
    template<typename T> struct Y {};
  };
  template<typename T> struct Z {
    X::Y<T> x;
    T::Y<T> y; // expected-error +{{}}
  };
  Z<X> z;
}

namespace dr122 { // dr122: yes
  template<typename T> void f();
  void g() { f<int>(); }
}

// dr123: na
// dr124: dup 201

// dr125: yes
struct dr125_A { struct dr125_B {}; };
dr125_A::dr125_B dr125_C();
namespace dr125_B { dr125_A dr125_C(); }
namespace dr125 {
  struct X {
    friend dr125_A::dr125_B (::dr125_C)(); // ok
    friend dr125_A (::dr125_B::dr125_C)(); // ok
    friend dr125_A::dr125_B::dr125_C(); // expected-error {{requires a type specifier}}
  };
}

namespace dr126 { // dr126: no
  struct C {};
  struct D : C {};
  struct E : private C { friend class A; friend class B; };
  struct F : protected C {};
  struct G : C {};
  struct H : D, G {};

  struct A {
    virtual void cp() throw(C*);
    virtual void dp() throw(C*);
    virtual void ep() throw(C*); // expected-note {{overridden}}
    virtual void fp() throw(C*); // expected-note {{overridden}}
    virtual void gp() throw(C*);
    virtual void hp() throw(C*); // expected-note {{overridden}}

    virtual void cr() throw(C&);
    virtual void dr() throw(C&);
    virtual void er() throw(C&); // expected-note {{overridden}}
    virtual void fr() throw(C&); // expected-note {{overridden}}
    virtual void gr() throw(C&);
    virtual void hr() throw(C&); // expected-note {{overridden}}

    virtual void pv() throw(void*); // expected-note {{overridden}}

#if __cplusplus >= 201103L
    virtual void np() throw(C*); // expected-note {{overridden}}
    virtual void npm() throw(int C::*); // expected-note {{overridden}}
    virtual void nr() throw(C&); // expected-note {{overridden}}
#endif

    virtual void ref1() throw(C *const&);
    virtual void ref2() throw(C *);

    virtual void v() throw(int);
    virtual void w() throw(const int);
    virtual void x() throw(int*);
    virtual void y() throw(const int*);
    virtual void z() throw(int); // expected-note {{overridden}}
  };
  struct B : A {
    virtual void cp() throw(C*);
    virtual void dp() throw(D*);
    virtual void ep() throw(E*); // expected-error {{more lax}}
    virtual void fp() throw(F*); // expected-error {{more lax}}
    virtual void gp() throw(G*);
    virtual void hp() throw(H*); // expected-error {{more lax}}

    virtual void cr() throw(C&);
    virtual void dr() throw(D&);
    virtual void er() throw(E&); // expected-error {{more lax}}
    virtual void fr() throw(F&); // expected-error {{more lax}}
    virtual void gr() throw(G&);
    virtual void hr() throw(H&); // expected-error {{more lax}}

    virtual void pv() throw(C*); // expected-error {{more lax}} FIXME: This is valid.

#if __cplusplus >= 201103L
    using nullptr_t = decltype(nullptr);
    virtual void np() throw(nullptr_t*); // expected-error {{more lax}} FIXME: This is valid.
    virtual void npm() throw(nullptr_t*); // expected-error {{more lax}} FIXME: This is valid.
    virtual void nr() throw(nullptr_t&); // expected-error {{more lax}} This is not.
#endif

    virtual void ref1() throw(D *const &);
    virtual void ref2() throw(D *);

    virtual void v() throw(const int);
    virtual void w() throw(int);
    virtual void x() throw(const int*); // FIXME: 'const int*' is not allowed by A::h.
    virtual void y() throw(int*); // ok
    virtual void z() throw(long); // expected-error {{more lax}}
  };
}

namespace dr127 { // dr127: yes
  __extension__ typedef __decltype(sizeof(0)) size_t;
  template<typename T> struct A {
    A() throw(int);
    void *operator new(size_t, const char * = 0);
    void operator delete(void *, const char *) { T::error; } // expected-error 2{{no members}}
    void operator delete(void *) { T::error; }
  };
  A<void> *p = new A<void>; // expected-note {{instantiat}}
  A<int> *q = new ("") A<int>; // expected-note {{instantiat}}
}

namespace dr128 { // dr128: yes
  enum E1 { e1 } x = e1;
  enum E2 { e2 } y = static_cast<E2>(x), z = static_cast<E2>(e1);
}

// dr129: dup 616
// dr130: na

namespace dr131 { // dr131: yes
  const char *a_with_\u0e8c = "\u0e8c";
  const char *b_with_\u0e8d = "\u0e8d";
  const char *c_with_\u0e8e = "\u0e8e";
#if __cplusplus < 201103L
  // expected-error@-4 {{expected ';'}} expected-error@-2 {{expected ';'}}
#endif
}

namespace dr132 { // dr132: no
  void f() {
    extern struct {} x; // ok
    extern struct S {} y; // FIXME: This is invalid.
  }
  static enum { E } e;
}

// dr133: dup 87
// dr134: na

namespace dr135 { // dr135: yes
  struct A {
    A f(A a) { return a; }
    friend A g(A a) { return a; }
    static A h(A a) { return a; }
  };
}

namespace dr136 { // dr136: no
  void f(int, int, int = 0);
  void g(int, int, int);
  struct A {
    // FIXME: These declarations of f, g, and h are invalid.
    friend void f(int, int = 0, int);
    friend void g(int, int, int = 0);
    friend void h(int, int, int = 0);
    friend void i(int, int, int = 0) {}
    friend void j(int, int, int = 0) {}
    operator int();
  };
  // FIXME: This declaration of i is invalid.
  void i(int, int, int);
  void q() {
    j(A(), A()); // ok, has default argument
  }
  // FIXME: Also test extern "C" friends and default arguments from other
  // namespaces?
}

namespace dr137 { // dr137: yes
  extern void *p;
  extern const void *cp;
  extern volatile void *vp;
  extern const volatile void *cvp;
  int *q = static_cast<int*>(p);
  int *qc = static_cast<int*>(cp); // expected-error {{casts away qualifiers}}
  int *qv = static_cast<int*>(vp); // expected-error {{casts away qualifiers}}
  int *qcv = static_cast<int*>(cvp); // expected-error {{casts away qualifiers}}
  const int *cq = static_cast<const int*>(p);
  const int *cqc = static_cast<const int*>(cp);
  const int *cqv = static_cast<const int*>(vp); // expected-error {{casts away qualifiers}}
  const int *cqcv = static_cast<const int*>(cvp); // expected-error {{casts away qualifiers}}
  const volatile int *cvq = static_cast<const volatile int*>(p);
  const volatile int *cvqc = static_cast<const volatile int*>(cp);
  const volatile int *cvqv = static_cast<const volatile int*>(vp);
  const volatile int *cvqcv = static_cast<const volatile int*>(cvp);
}

namespace dr139 { // dr139: yes
  namespace example1 {
    typedef int f; // expected-note {{previous}}
    struct A {
      friend void f(A &); // expected-error {{different kind of symbol}}
    };
  }

  namespace example2 {
    typedef int f;
    namespace N {
      struct A {
        friend void f(A &);
        operator int();
        void g(A a) { int i = f(a); } // ok, f is typedef not friend function
      };
    }
  }
}

namespace dr140 { // dr140: yes
  void f(int *const) {} // expected-note {{previous}}
  void f(int[3]) {} // expected-error {{redefinition}}
  void g(const int);
  void g(int n) { n = 2; }
}

namespace dr141 { // dr141: yes
  template<typename T> void f();
  template<typename T> struct S { int n; };
  struct A : S<int> {
    template<typename T> void f();
    template<typename T> struct S {};
  } a;
  struct B : S<int> {} b;
  void g() {
    a.f<int>();
    (void)a.S<int>::n; // expected-error {{no member named 'n'}}
#if __cplusplus < 201103L
    // expected-error@-2 {{ambiguous}}
    // expected-note@-11 {{lookup from the current scope}}
    // expected-note@-9 {{lookup in the object type}}
#endif
    b.f<int>(); // expected-error {{no member}} expected-error +{{}}
    (void)b.S<int>::n;
  }
  template<typename T> struct C {
    T t;
    void g() {
      t.f<int>(); // expected-error {{use 'template'}}
    }
    void h() {
      (void)t.S<int>::n; // ok
    }
    void i() {
      (void)t.S<int>(); // ok!
    }
  };
  void h() { C<B>().h(); } // ok
  struct X {
    template<typename T> void S();
  };
  void i() { C<X>().i(); } // ok!!
}

namespace dr142 { // dr142: yes
  class B { // expected-note +{{here}}
  public:
    int mi; // expected-note +{{here}}
    static int si; // expected-note +{{here}}
  };
  class D : private B { // expected-note +{{here}}
  };
  class DD : public D {
    void f();
  };
  void DD::f() {
    mi = 3; // expected-error {{private base class}} expected-error {{private member}}
    si = 3; // expected-error {{private member}}
    B b_old; // expected-error {{private member}}
    dr142::B b;
    b.mi = 3;
    b.si = 3;
    B::si = 3; // expected-error {{private member}}
    dr142::B::si = 3;
    B *bp1_old = this; // expected-error {{private member}} expected-error {{private base class}}
    dr142::B *bp1 = this; // expected-error {{private base class}}
    B *bp2_old = (B*)this; // expected-error 2{{private member}}
    dr142::B *bp2 = (dr142::B*)this;
    bp2->mi = 3;
  }
}

namespace dr143 { // dr143: yes
  namespace A { struct X; }
  namespace B { void f(A::X); }
  namespace A {
    struct X { friend void B::f(X); };
  }
  void g(A::X x) {
    f(x); // expected-error {{undeclared identifier 'f'}}
  }
}

namespace dr145 { // dr145: yes
  void f(bool b) {
    ++b; // expected-warning {{deprecated}}
    b++; // expected-warning {{deprecated}}
  }
}

namespace dr147 { // dr147: no
  namespace example1 {
    template<typename> struct A {
      template<typename T> A(T);
    };
    // FIXME: This appears to be valid, and EDG and G++ accept.
    template<> template<> A<int>::A<int>(int) {} // expected-error {{out-of-line constructor for 'A' cannot have template arguments}}
  }
  namespace example2 {
    struct A { A(); };
    struct B : A { B(); };
    A::A a1; // expected-error {{is a constructor}}
    B::A a2;
  }
  namespace example3 {
    template<typename> struct A {
      template<typename T> A(T);
      static A a;
    };
    template<> A<int>::A<int>(A<int>::a); // expected-error {{is a constructor}}
  }
}

namespace dr148 { // dr148: yes
  struct A { int A::*p; };
  int check1[__is_pod(int(A::*)) ? 1 : -1];
  int check2[__is_pod(A) ? 1 : -1];
}

// dr149: na
