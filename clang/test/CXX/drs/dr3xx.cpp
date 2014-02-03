// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr300 { // dr300: yes
  template<typename R, typename A> void f(R (&)(A)) {}
  int g(int);
  void h() { f(g); }
}

namespace dr301 { // dr301: yes
  // see also dr38
  struct S;
  template<typename T> void operator+(T, T);
  void operator-(S, S);

  void f() {
    bool a = (void(*)(S, S))operator+<S> <
             (void(*)(S, S))operator+<S>;
    bool b = (void(*)(S, S))operator- <
             (void(*)(S, S))operator-;
    bool c = (void(*)(S, S))operator+ <
             (void(*)(S, S))operator-; // expected-error {{expected '>'}}
  }

  template<typename T> void f() {
    typename T::template operator+<int> a; // expected-error {{typename specifier refers to a non-type template}} expected-error +{{}}
    // FIXME: This shouldn't say (null).
    class T::template operator+<int> b; // expected-error {{identifier followed by '<' indicates a class template specialization but (null) refers to a function template}}
    enum T::template operator+<int> c; // expected-error {{expected identifier}} expected-error {{does not declare anything}}
    enum T::template operator+<int>::E d; // expected-error {{qualified name refers into a specialization of function template 'T::template operator +'}} expected-error {{forward reference}}
    enum T::template X<int>::E e;
    T::template operator+<int>::foobar(); // expected-error {{qualified name refers into a specialization of function template 'T::template operator +'}}
    T::template operator+<int>(0); // ok
  }

  template<typename T> class operator&<T*> {}; // expected-error +{{}}
  template<typename T> class T::operator& {}; // expected-error +{{}}
  template<typename T> class S::operator&<T*> {}; // expected-error +{{}}
}

namespace dr302 { // dr302: yes
  struct A { A(); ~A(); };
#if __cplusplus < 201103L
  struct B { // expected-error {{implicit default constructor for 'dr302::B' must explicitly initialize the const member 'n'}}
    const int n; // expected-note {{declared here}}
    A a;
  } b = B(); // expected-note {{first required here}}
  // Trivial default constructor C::C() is not called here.
  struct C {
    const int n;
  } c = C();
#else
  struct B {
    const int n; // expected-note {{deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
    A a;
  } b = B(); // expected-error {{call to implicitly-deleted default constructor}}
  // C::C() is called here, because even though it's trivial, it's deleted.
  struct C {
    const int n; // expected-note {{deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
  } c = C(); // expected-error {{call to implicitly-deleted default constructor}}
  struct D {
    const int n = 0;
  } d = D();
#endif
}

// dr303: na

namespace dr304 { // dr304: yes
  typedef int &a;
  int n = a(); // expected-error {{requires an initializer}}

  struct S { int &b; };
  int m = S().b;
#if __cplusplus < 201103L
  // expected-error@-3 {{requires an initializer}}
  // expected-note@-3 {{in value-initialization}}
#else
  // expected-error@-5 {{deleted}}
  // expected-note@-7 {{reference}}
#endif
}

namespace dr305 { // dr305: yes
  struct A {
    typedef A C;
  };
  void f(A *a) {
    struct A {};
    a->~A();
    a->~C();
  }
  typedef A B;
  void g(B *b) {
    b->~B();
    b->~C();
  }
  void h(B *b) {
    struct B {}; // expected-note {{declared here}}
    b->~B(); // expected-error {{does not match}}
  }

  template<typename T> struct X {};
  void i(X<int>* x) {
    struct X {};
    x->~X<int>();
    x->~X();
    x->~X<char>(); // expected-error {{no member named}}
  }

#if __cplusplus >= 201103L
  struct Y {
    template<typename T> using T1 = Y;
  };
  template<typename T> using T2 = Y;
  void j(Y *y) {
    y->~T1<int>();
    y->~T2<int>();
  }
  struct Z {
    template<typename T> using T2 = T;
  };
  void k(Z *z) {
    // FIXME: This diagnostic is terrible.
    z->~T1<int>(); // expected-error {{'T1' following the 'template' keyword does not refer to a template}} expected-error +{{}}
    z->~T2<int>(); // expected-error {{no member named '~int'}}
    z->~T2<Z>();
  }
#endif
}

namespace dr306 { // dr306: no
  // FIXME: dup 39
  // FIXME: This should be accepted.
  struct A { struct B {}; }; // expected-note 2{{member}}
  struct C { typedef A::B B; }; // expected-note {{member}}
  struct D : A, A::B, C {};
  D::B b; // expected-error {{found in multiple base classes of different types}}
}

// dr307: na

namespace dr308 { // dr308: yes
  // This is mostly an ABI library issue.
  struct A {};
  struct B : A {};
  struct C : A {};
  struct D : B, C {};
  void f() {
    try {
      throw D();
    } catch (const A&) {
      // unreachable
    } catch (const B&) {
      // get here instead
    }
  }
}

// dr309: dup 485

namespace dr311 { // dr311: yes
  namespace X { namespace Y {} }
  namespace X::Y {} // expected-error {{must define each namespace separately}}
  namespace X {
    namespace X::Y {} // expected-error {{must define each namespace separately}}
  }
  // FIXME: The diagnostics here are not very good.
  namespace ::dr311::X {} // expected-error 2+{{}} // expected-warning {{extra qual}}
}

// dr312: dup 616

namespace dr313 { // dr313: dup 299 c++11
  struct A { operator int() const; };
  int *p = new int[A()];
#if __cplusplus < 201103L
  // FIXME: should this be available in c++98 mode? expected-error@-2 {{extension}}
#endif
}

// dr315: na
// dr316: sup 1004

namespace dr317 { // dr317: no
  void f() {}
  inline void f(); // FIXME: ill-formed

  int g();
  int n = g();
  inline int g() { return 0; }

  int h();
  int m = h();
  int h() { return 0; }
  inline int h(); // FIXME: ill-formed
}

namespace dr318 { // dr318: sup 1310
  struct A {};
  struct A::A a;
}

namespace dr319 { // dr319: no
  // FIXME: dup dr389
  // FIXME: We don't have a diagnostic for a name with linkage
  //        having a type without linkage.
  typedef struct {
    int i;
  } *ps;
  extern "C" void f(ps);
  void g(ps); // FIXME: ill-formed, type 'ps' has no linkage

  static enum { e } a1;
  enum { e2 } a2; // FIXME: ill-formed, enum type has no linkage

  enum { n1 = 1u };
  typedef int (*pa)[n1];
  pa parr; // ok, type has linkage despite using 'n1'

  template<typename> struct X {};

  void f() {
    struct A { int n; };
    extern A a; // FIXME: ill-formed
    X<A> xa;

    typedef A B;
    extern B b; // FIXME: ill-formed
    X<B> xb;

    const int n = 1;
    typedef int (*C)[n];
    extern C c; // ok
    X<C> xc;
  }
#if __cplusplus < 201103L
  // expected-error@-12 {{uses local type 'A'}}
  // expected-error@-9 {{uses local type 'A'}}
#endif
}

namespace dr320 { // dr320: yes
#if __cplusplus >= 201103L
  struct X {
    constexpr X() {}
    constexpr X(const X &x) : copies(x.copies + 1) {}
    unsigned copies = 0;
  };
  constexpr X f(X x) { return x; }
  constexpr unsigned g(X x) { return x.copies; }
  static_assert(f(X()).copies == g(X()) + 1, "expected one extra copy for return value");
#endif
}

namespace dr321 { // dr321: dup 557
  namespace N {
    template<int> struct A {
      template<int> struct B;
    };
    template<> template<> struct A<0>::B<0>;
    void f(A<0>::B<0>);
  }
  template<> template<> struct N::A<0>::B<0> {};

  template<typename T> void g(T t) { f(t); }
  template void g(N::A<0>::B<0>);

  namespace N {
    template<typename> struct I { friend bool operator==(const I&, const I&); };
  }
  N::I<int> i, j;
  bool x = i == j;
}

namespace dr322 { // dr322: yes
  struct A {
    template<typename T> operator T&();
  } a;
  int &r = static_cast<int&>(a);
  int &s = a;
}

// dr323: no

namespace dr324 { // dr324: yes
  struct S { int n : 1; } s; // expected-note 3{{bit-field is declared here}}
  int &a = s.n; // expected-error {{non-const reference cannot bind to bit-field}}
  int *b = &s.n; // expected-error {{address of bit-field}}
  int &c = (s.n = 0); // expected-error {{non-const reference cannot bind to bit-field}}
  int *d = &(s.n = 0); // expected-error {{address of bit-field}}
  int &e = true ? s.n : s.n; // expected-error {{non-const reference cannot bind to bit-field}}
  int *f = &(true ? s.n : s.n); // expected-error {{address of bit-field}}
  int &g = (void(), s.n); // expected-error {{non-const reference cannot bind to bit-field}}
  int *h = &(void(), s.n); // expected-error {{address of bit-field}}
}

namespace dr326 { // dr326: yes
  struct S {};
  int test[__is_trivially_constructible(S, const S&) ? 1 : -1];
}

namespace dr327 { // dr327: dup 538
  struct A;
  class A {};

  class B;
  struct B {};
}

namespace dr328 { // dr328: yes
  struct A; // expected-note 3{{forward declaration}}
  struct B { A a; }; // expected-error {{incomplete}}
  template<typename> struct C { A a; }; // expected-error {{incomplete}}
  A *p = new A[0]; // expected-error {{incomplete}}
}

namespace dr329 { // dr329: no
  // FIXME: The C++98 behavior here is right, the C++11-onwards behavior
  // is wrong.
  struct B {};
  template<typename T> struct A : B {
    friend void f(A a) { g(a); }
    friend void h(A a) { g(a); } // expected-error {{undeclared}}
    friend void i(B b) {}
  };
  A<int> a;
  A<char> b;
#if __cplusplus < 201103L
  // expected-error@-5 {{redefinition}} expected-note@-5 {{previous}}
  // expected-note@-3 {{instantiation}}
#endif

  void test() {
    h(a); // expected-note {{instantiation}}
    i(a);
    i(b);
  }
}
