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
