// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr500 { // dr500: dup 372
  class D;
  class A {
    class B;
    class C;
    friend class D;
  };
  class A::B {};
  class A::C : public A::B {};
  class D : public A::B {};
}

namespace dr501 { // dr501: yes
  struct A {
    friend void f() {}
    void g() {
      void (*p)() = &f; // expected-error {{undeclared identifier}}
    }
  };
}

namespace dr502 { // dr502: yes
  struct Q {};
  template<typename T> struct A {
    enum E { e = 1 };
    void q1() { f(e); }
    void q2() { Q arr[sizeof(E)]; f(arr); }
    void q3() { Q arr[e]; f(arr); }
    void sanity() { Q arr[1]; f(arr); } // expected-error {{undeclared identifier 'f'}}
  };
  int f(A<int>::E);
  template<int N> int f(Q (&)[N]);
  template struct A<int>;
}

namespace dr505 { // dr505: yes
  const char *exts = "\e\(\{\[\%"; // expected-error 5{{use of non-standard escape}}
  const char *unknown = "\Q"; // expected-error {{unknown escape sequence}}
}

namespace dr506 { // dr506: yes
  struct NonPod { ~NonPod(); };
  void f(...);
  void g(NonPod np) { f(np); } // expected-error {{cannot pass}}
}

// FIXME: Add tests here once DR260 is resolved.
// dr507: dup 260

// dr508: na
// dr509: na
// dr510: na

namespace dr512 { // dr512: yes
  struct A {
    A(int);
  };
  union U { A a; };
#if __cplusplus < 201103L
  // expected-error@-2 {{has a non-trivial constructor}}
  // expected-note@-6 {{no default constructor}}
  // expected-note@-6 {{suppressed by user-declared constructor}}
#endif
}

// dr513: na

namespace dr514 { // dr514: yes
  namespace A { extern int x, y; }
  int A::x = y;
}

namespace dr515 { // dr515: sup 1017
  // FIXME: dr1017 reverses the wording of dr515, but the current draft has
  // dr515's wording, with a different fix for dr1017.

  struct X { int n; };
  template<typename T> struct Y : T {
    int f() { return X::n; }
  };
  int k = Y<X>().f();

  struct A { int a; };
  struct B { void f() { int k = sizeof(A::a); } };
#if __cplusplus < 201103L
  // expected-error@-2 {{invalid use of non-static data member}}
#endif
}

// dr516: na

namespace dr517 { // dr517: no
  // This is NDR, but we should diagnose it anyway.
  template<typename T> struct S {};
  template<typename T> int v = 0; // expected-error 0-1{{extension}}

  template struct S<int*>;
  template int v<int*>;

  S<char&> s;
  int k = v<char&>;

  // FIXME: These are both ill-formed.
  template<typename T> struct S<T*> {};
  template<typename T> int v<T*> = 0; // expected-error 0-1{{extension}}

  // FIXME: These are both ill-formed.
  template<typename T> struct S<T&> {};
  template<typename T> int v<T&> = 0; // expected-error 0-1{{extension}}
}

namespace dr518 { // dr518: yes c++11
  enum E { e, };
#if __cplusplus < 201103L
  // expected-error@-2 {{C++11 extension}}
#endif
}

namespace dr519 { // dr519: yes
// FIXME: Add a codegen test.
#if __cplusplus >= 201103L
#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
  int test[fold((int*)(void*)0) ? -1 : 1];
#undef fold
#endif
}

// dr520: na

// dr521: no
// FIXME: The wording here is broken. It's not reasonable to expect a
// diagnostic here. Once the relevant DR gets a number, mark this as a dup.

namespace dr522 { // dr522: yes
  struct S {};
  template<typename T> void b1(volatile T &);
  template<typename T> void b2(volatile T * const *);
  template<typename T> void b2(volatile T * const S::*);
  template<typename T> void b2(volatile T * const S::* const *);
  // FIXME: This diagnostic isn't very good. The problem is not substitution failure.
  template<typename T> void b2a(volatile T *S::* const *); // expected-note {{substitution failure}}

  template<typename T> struct Base {};
  struct Derived : Base<int> {};
  template<typename T> void b3(Base<T>);
  template<typename T> void b3(Base<T> *);

  void test(int n, const int cn, int **p, int *S::*pm) {
    int *a[3], *S::*am[3]; 
    const Derived cd = Derived();
    Derived d[3];

    b1(n);
    b1(cn);
    b2(p);
    b2(pm);
    b2(a);
    b2(am);
    b2a(am); // expected-error {{no matching function}}
    b3(d);
    b3(cd);
  }
}

namespace dr524 { // dr524: yes
  template<typename T> void f(T a, T b) { operator+(a, b); } // expected-error {{call}}

  struct S {};
  void operator+(S, S);
  template void f(S, S);

  namespace N { struct S {}; }
  void operator+(N::S, N::S); // expected-note {{should be declared}}
  template void f(N::S, N::S); // expected-note {{instantiation}}
}

namespace dr525 { // dr525: yes
  namespace before {
    // Note, the example was correct prior to the change; instantiation is
    // required for cases like this:
    template <class T> struct D { operator T*(); };
    void g(D<double> ppp) {
      delete ppp;
    }
  }
  namespace after {
    template <class T> struct D { typename T::error e; }; // expected-error {{prior to '::'}}
    void g(D<double> *ppp) {
      delete ppp; // expected-note {{instantiation of}}
    }
  }
}

namespace dr526 { // dr526: yes
  template<int> struct S {};
  template<int N> void f1(S<N> s);
  template<int N> void f2(S<(N)> s); // expected-note {{couldn't infer}}
  template<int N> void f3(S<+N> s); // expected-note {{couldn't infer}}
  template<int N> void g1(int (&)[N]);
  template<int N> void g2(int (&)[(N)]); // expected-note {{couldn't infer}}
  template<int N> void g3(int (&)[+N]); // expected-note {{couldn't infer}}

  void test(int (&a)[3], S<3> s) {
    f1(s);
    f2(s); // expected-error {{no matching}}
    f3(s); // expected-error {{no matching}}
    g1(a);
    g2(a); // expected-error {{no matching}}
    g3(a); // expected-error {{no matching}}
  }

  template<int N> struct X {
    typedef int type;
    X<N>::type v1;
    X<(N)>::type v2; // expected-error {{missing 'typename'}}
    X<+N>::type v3; // expected-error {{missing 'typename'}}
  };
}

namespace dr527 { // dr527: na
  // This DR is meaningless. It removes a required diagnostic from the case
  // where a not-externally-visible object is odr-used but not defined, which
  // requires a diagnostic for a different reason.
  extern struct { int x; } a; // FIXME: We should reject this, per dr389.
  static struct { int x; } b;
  extern "C" struct { int x; } c;
  namespace { extern struct { int x; } d; }
  typedef struct { int x; } *P;
  struct E { static P e; }; // FIXME: We should reject this, per dr389.
  namespace { struct F { static P f; }; }

  int ax = a.x, bx = b.x, cx = c.x, dx = d.x, ex = E::e->x, fx = F::f->x;
}

namespace dr530 { // dr530: yes
  template<int*> struct S { enum { N = 1 }; };
  template<void(*)()> struct T { enum { N = 1 }; };
  int n;
  void f();
  int a[S<&n>::N];
  int b[T<&f>::N];
}

namespace dr531 { // dr531: partial
  namespace good {
    template<typename T> struct A {
      void f(T) { T::error; }
      template<typename U> void g(T, U) { T::error; }
      struct B { typename T::error error; };
      template<typename U> struct C { typename T::error error; };
      static T n;
    };
    template<typename T> T A<T>::n = T::error;

    template<> void A<int>::f(int) {}
    template<> template<typename U> void A<int>::g(int, U) {}
    template<> struct A<int>::B {};
    template<> template<typename U> struct A<int>::C {};
    template<> int A<int>::n = 0;

    void use(A<int> a) {
      a.f(a.n);
      a.g(0, 0);
      A<int>::B b;
      A<int>::C<int> c;
    }

    template<> struct A<char> {
      void f(char);
      template<typename U> void g(char, U);
      struct B;
      template<typename U> struct C;
      static char n;
    };

    void A<char>::f(char) {}
    template<typename U> void A<char>::g(char, U) {}
    struct A<char>::B {};
    template<typename U> struct A<char>::C {};
    char A<char>::n = 0;
  }

  namespace bad {
    template<typename T> struct A {
      void f(T) { T::error; }
      template<typename U> void g(T, U) { T::error; }
      struct B { typename T::error error; };
      template<typename U> struct C { typename T::error error; }; // expected-note {{here}}
      static T n;
    };
    template<typename T> T A<T>::n = T::error;

    void A<int>::f(int) {} // expected-error {{requires 'template<>'}}
    template<typename U> void A<int>::g(int, U) {} // expected-error {{should be empty}}
    struct A<int>::B {}; // expected-error {{requires 'template<>'}}
    template<typename U> struct A<int>::C {}; // expected-error {{should be empty}} expected-error {{different kind of symbol}}
    int A<int>::n = 0; // expected-error {{requires 'template<>'}}

    template<> struct A<char> { // expected-note 2{{here}}
      void f(char);
      template<typename U> void g(char, U);
      struct B; // expected-note {{here}}
      template<typename U> struct C;
      static char n;
    };

    template<> void A<char>::f(char) {} // expected-error {{no function template matches}}
    // FIXME: This is ill-formed; -pedantic-errors should reject.
    template<> template<typename U> void A<char>::g(char, U) {} // expected-warning {{extraneous template parameter list}}
    template<> struct A<char>::B {}; // expected-error {{extraneous 'template<>'}} expected-error {{does not specialize}}
    // FIXME: This is ill-formed; -pedantic-errors should reject.
    template<> template<typename U> struct A<char>::C {}; // expected-warning {{extraneous template parameter list}}
    template<> char A<char>::n = 0; // expected-error {{extraneous 'template<>'}}
  }

  namespace nested {
    template<typename T> struct A {
      template<typename U> struct B;
    };
    template<> template<typename U> struct A<int>::B {
      void f();
      void g();
      template<typename V> void h();
      template<typename V> void i();
    };
    template<> template<typename U> void A<int>::B<U>::f() {}
    template<typename U> void A<int>::B<U>::g() {} // expected-error {{should be empty}}

    template<> template<typename U> template<typename V> void A<int>::B<U>::h() {}
    template<typename U> template<typename V> void A<int>::B<U>::i() {} // expected-error {{should be empty}}

    template<> template<> void A<int>::B<int>::f() {}
    template<> template<> template<typename V> void A<int>::B<int>::h() {}
    template<> template<> template<> void A<int>::B<int>::h<int>() {}

    template<> void A<int>::B<char>::f() {} // expected-error {{requires 'template<>'}}
    template<> template<typename V> void A<int>::B<char>::h() {} // expected-error {{should be empty}}
  }
}

// PR8130
namespace dr532 { // dr532: 3.5
  struct A { };

  template<class T> struct B {
    template<class R> int &operator*(R&);
  };

  template<class T, class R> float &operator*(T&, R&);
  void test() {
    A a;
    B<A> b;
    int &ir = b * a;
  }
}

// dr533: na

namespace dr534 { // dr534: yes
  struct S {};
  template<typename T> void operator+(S, T);
  template<typename T> void operator+<T*>(S, T*) {} // expected-error {{function template partial spec}}
}

namespace dr535 { // dr535: yes
  class X { private: X(const X&); };
  struct A {
    X x;
    template<typename T> A(T&);
  };
  struct B : A {
    X y;
    B(volatile A&);
  };

  extern A a1;
  A a2(a1); // ok, uses constructor template

  extern volatile B b1;
  B b2(b1); // ok, uses converting constructor

  void f() { throw a1; }

#if __cplusplus >= 201103L
  struct C {
    constexpr C() : n(0) {}
    template<typename T> constexpr C(T&t) : n(t.n == 0 ? throw 0 : 0) {}
    int n;
  };
  constexpr C c() { return C(); }
  // ok, copy is elided
  constexpr C x = c();
#endif
}

// dr537: na
// dr538: na

// dr539: yes
const dr539( // expected-error {{requires a type specifier}}
    const a) { // expected-error {{unknown type name 'a'}}
  const b; // expected-error {{requires a type specifier}}
  new const; // expected-error {{expected a type}}
  try {} catch (const n) {} // expected-error {{unknown type name 'n'}}
  try {} catch (const) {} // expected-error {{expected a type}}
  if (const n = 0) {} // expected-error {{requires a type specifier}}
  switch (const n = 0) {} // expected-error {{requires a type specifier}}
  while (const n = 0) {} // expected-error {{requires a type specifier}}
  for (const n = 0; // expected-error {{requires a type specifier}}
       const m = 0; ) {} // expected-error {{requires a type specifier}}
  sizeof(const); // expected-error {{requires a type specifier}}
  struct S {
    const n; // expected-error {{requires a type specifier}}
    operator const(); // expected-error {{expected a type}}
  };
#if __cplusplus >= 201103L
  int arr[3];
  // FIXME: The extra braces here are to avoid the parser getting too
  // badly confused when recovering here. We should fix this recovery.
  { for (const n // expected-error {{unknown type name 'n'}} expected-note {{}}
         : arr) ; {} } // expected-error +{{}}
  (void) [](const) {}; // expected-error {{requires a type specifier}}
  (void) [](const n) {}; // expected-error {{unknown type name 'n'}}
  enum E : const {}; // expected-error {{expected a type}}
  using T = const; // expected-error {{expected a type}}
  auto f() -> const; // expected-error {{expected a type}}
#endif
}

namespace dr540 { // dr540: yes
  typedef int &a;
  typedef const a &a; // expected-warning {{has no effect}}
  typedef const int &b;
  typedef b &b;
  typedef const a &c; // expected-note {{previous}} expected-warning {{has no effect}}
  typedef const b &c; // expected-error {{different}} expected-warning {{has no effect}}
}

namespace dr541 { // dr541: yes
  template<int> struct X { typedef int type; };
  template<typename T> struct S {
    int f(T);

    int g(int);
    T g(bool);

    int h();
    int h(T);

    void x() {
      // These are type-dependent expressions, even though we could
      // determine that all calls have type 'int'.
      X<sizeof(f(0))>::type a; // expected-error +{{}}
      X<sizeof(g(0))>::type b; // expected-error +{{}}
      X<sizeof(h(0))>::type b; // expected-error +{{}}

      typename X<sizeof(f(0))>::type a;
      typename X<sizeof(h(0))>::type b;
    }
  };
}

namespace dr542 { // dr542: yes
#if __cplusplus >= 201103L
  struct A { A() = delete; int n; };
  A a[32] = {}; // ok, constructor not called

  struct B {
    int n;
  private:
    B() = default;
  };
  B b[32] = {}; // ok, constructor not called
#endif
}

namespace dr543 { // dr543: yes
  // In C++98+DR543, this is valid because value-initialization doesn't call a
  // trivial default constructor, so we never notice that defining the
  // constructor would be ill-formed.
  //
  // In C++11+DR543, this is ill-formed, because the default constructor is
  // deleted, and value-initialization *does* call a deleted default
  // constructor, even if it is trivial.
  struct A {
    const int n;
  };
  A a = A();
#if __cplusplus >= 201103L
  // expected-error@-2 {{deleted}}
  // expected-note@-5 {{would not be initialized}}
#endif
}

namespace dr544 { // dr544: yes
  int *n;

  template<class T> struct A { int n; };
  template<class T> struct B : A<T> { int get(); };
  template<> int B<int>::get() { return n; }
  int k = B<int>().get();
}

namespace dr546 { // dr546: yes
  template<typename T> struct A { void f(); };
  template struct A<int>;
  template<typename T> void A<T>::f() { T::error; }
}

#if !defined(_M_IX86) // This is incompatible to attribute(thiscall).
namespace dr547 { // d547: yes
  template<typename T> struct X;
  template<typename T> struct X<T() const> {};
  template<typename T, typename C> X<T> f(T C::*) { return X<T>(); }

  struct S { void f() const; };
  X<void() const> x = f(&S::f);
}
#endif

namespace dr548 { // dr548: dup 482
  template<typename T> struct S {};
  template<typename T> void f() {}
  template struct dr548::S<int>;
  template void dr548::f<int>();
}
