// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr400 { // dr400: yes
  struct A { int a; struct a {}; }; // expected-note 2{{conflicting}} expected-note {{ambiguous}}
  struct B { int a; struct a {}; }; // expected-note 2{{target}} expected-note {{ambiguous}}
  struct C : A, B { using A::a; struct a b; };
  struct D : A, B { using A::a; using B::a; struct a b; }; // expected-error 2{{conflicts}}
  struct E : A, B { struct a b; }; // expected-error {{found in multiple base classes}}
}

namespace dr401 { // dr401: yes
  template<class T, class U = typename T::type> class A : public T {}; // expected-error {{protected}} expected-error 2{{private}}

  class B {
  protected:
    typedef int type; // expected-note {{protected}}
  };

  class C {
    typedef int type; // expected-note {{private}}
    friend class A<C>; // expected-note {{default argument}}
  };

  class D {
    typedef int type; // expected-note {{private}}
    friend class A<D, int>;
  };

  A<B> *b; // expected-note {{default argument}}
  // FIXME: We're missing the "in instantiation of" note for the default
  // argument here.
  A<D> *d;

  struct E {
    template<class T, class U = typename T::type> class A : public T {};
  };
  class F {
    typedef int type;
    friend class E;
  };
  E::A<F> eaf; // ok, default argument is in befriended context

  // FIXME: Why do we get different diagnostics in C++11 onwards here? We seem
  // to not treat the default template argument as a SFINAE context in C++98.
  template<class T, class U = typename T::type> void f(T) {}
  void g(B b) { f(b); }
#if __cplusplus < 201103L
  // expected-error@-3 0-1{{extension}} expected-error@-3 {{protected}} expected-note@-3 {{instantiation}}
  // expected-note@-3 {{substituting}}
#else
  // expected-error@-5 {{no matching}} expected-note@-6 {{protected}}
#endif
}

namespace dr403 { // dr403: yes
  namespace A {
    struct S {};
    int f(void*);
  }
  template<typename T> struct X {};
  typedef struct X<A::S>::X XS;
  XS *p;
  int k = f(p); // ok, finds A::f, even though type XS is a typedef-name
                // referring to an elaborated-type-specifier naming a
                // injected-class-name, which is about as far from a
                // template-id as we can make it.
}

// dr404: na
// (NB: also sup 594)

namespace dr406 { // dr406: yes
  typedef struct {
    static int n; // expected-error {{static data member 'n' not allowed in anonymous struct}}
  } A;
}

namespace dr407 { // dr407: no
  struct S;
  typedef struct S S;
  void f() {
    struct S *p;
    {
      typedef struct S S; // expected-note {{here}}
      struct S *p; // expected-error {{refers to a typedef}}
    }
  }
  struct S {};

  namespace UsingDir {
    namespace A {
      struct S {}; // expected-note {{found}}
    }
    namespace B {
      typedef int S; // expected-note {{found}}
    }
    namespace C {
      using namespace A;
      using namespace B;
      struct S s; // expected-error {{ambiguous}}
    }
    namespace D {
      // FIXME: This is valid.
      using A::S;
      typedef struct S S; // expected-note {{here}}
      struct S s; // expected-error {{refers to a typedef}}
    }
    namespace E {
      // FIXME: The standard doesn't say whether this is valid.
      typedef A::S S;
      using A::S;
      struct S s;
    }
    namespace F {
      typedef A::S S; // expected-note {{here}}
    }
    // FIXME: The standard doesn't say what to do in these cases, but
    // our behavior should not depend on the order of the using-directives.
    namespace G {
      using namespace A;
      using namespace F;
      struct S s;
    }
    namespace H {
      using namespace F;
      using namespace A;
      struct S s; // expected-error {{refers to a typedef}}
    }
  }
}

namespace dr408 { // dr408: 3.4
  template<int N> void g() { int arr[N != 1 ? 1 : -1]; }
  template<> void g<2>() { }

  template<typename T> struct S {
    static int i[];
    void f();
  };
  template<typename T> int S<T>::i[] = { 1 };

  template<typename T> void S<T>::f() {
    g<sizeof (i) / sizeof (int)>();
  }
  template<> int S<int>::i[] = { 1, 2 };
  template void S<int>::f(); // uses g<2>(), not g<1>().


  template<typename T> struct R {
    static int arr[];
    void f();
  };
  template<typename T> int R<T>::arr[1];
  template<typename T> void R<T>::f() {
    int arr[sizeof(arr) != sizeof(int) ? 1 : -1];
  }
  template<> int R<int>::arr[2];
  template void R<int>::f();
}

namespace dr409 { // dr409: yes
  template<typename T> struct A {
    typedef int B;
    B b1;
    A::B b2;
    A<T>::B b3;
    A<T*>::B b4; // expected-error {{missing 'typename'}}
  };
}

namespace dr410 { // dr410: no
  template<class T> void f(T);
  void g(int);
  namespace M {
    template<class T> void h(T);
    template<class T> void i(T);
    struct A {
      friend void f<>(int);
      friend void h<>(int);
      friend void g(int);
      template<class T> void i(T);
      friend void i<>(int);
    private:
      static void z(); // expected-note {{private}}
    };

    template<> void h(int) { A::z(); }
    // FIXME: This should be ill-formed. The member A::i<> is befriended,
    // not this function.
    template<> void i(int) { A::z(); }
  }
  template<> void f(int) { M::A::z(); }
  void g(int) { M::A::z(); } // expected-error {{private}}
}

// dr412 is in its own file.

namespace dr413 { // dr413: yes
  struct S {
    int a;
    int : 17;
    int b;
  };
  S s = { 1, 2, 3 }; // expected-error {{excess elements}}
}

namespace dr414 { // dr414: dup 305
  struct X {};
  void f() {
    X x;
    struct X {};
    x.~X();
  }
}

namespace dr415 { // dr415: yes
  template<typename T> void f(T, ...) { T::error; }
  void f(int, int);
  void g() { f(0, 0); } // ok
}

namespace dr416 { // dr416: yes
  extern struct A a;
  int &operator+(const A&, const A&);
  int &k = a + a;
  struct A { float &operator+(A&); };
  float &f = a + a;
}

namespace dr417 { // dr417: no
  struct A;
  struct dr417::A {}; // expected-warning {{extra qualification}}
  struct B { struct X; };
  struct C : B {};
  struct C::X {}; // expected-error {{no struct named 'X' in 'dr417::C'}}
  struct B::X { struct Y; };
  struct C::X::Y {}; // ok!
  namespace N {
    struct D;
    struct E;
    struct F;
    struct H;
  }
  // FIXME: This is ill-formed.
  using N::D;
  struct dr417::D {}; // expected-warning {{extra qualification}}
  using namespace N;
  struct dr417::E {}; // expected-warning {{extra qualification}} expected-error {{no struct named 'E'}}
  struct N::F {};
  struct G;
  using N::H;
  namespace M {
    struct dr417::G {}; // expected-error {{namespace 'M' does not enclose}}
    struct dr417::H {}; // expected-error {{namespace 'M' does not enclose}}
  }
}

namespace dr420 { // dr420: yes
  template<typename T> struct ptr {
    T *operator->() const;
    T &operator*() const;
  };
  template<typename T, typename P> void test(P p) {
    p->~T();
    p->T::~T();
    (*p).~T();
    (*p).T::~T();
  }
  struct X {};
  template void test<int>(int*);
  template void test<int>(ptr<int>);
  template void test<X>(X*);
  template void test<X>(ptr<X>);

  template<typename T>
  void test2(T p) {
    p->template Y<int>::~Y<int>();
    p->~Y<int>();
    // FIXME: This is ill-formed, but this diagnostic is terrible. We should
    // reject this in the parser.
    p->template ~Y<int>(); // expected-error 2{{no member named '~typename Y<int>'}}
  }
  template<typename T> struct Y {};
  template void test2(Y<int>*); // expected-note {{instantiation}}
  template void test2(ptr<Y<int> >); // expected-note {{instantiation}}

  void test3(int *p, ptr<int> q) {
    typedef int Int;
    p->~Int();
    q->~Int();
    p->Int::~Int();
    q->Int::~Int();
  }

#if __cplusplus >= 201103L
  template<typename T> using id = T;
  struct A { template<typename T> using id = T; };
  void test4(int *p, ptr<int> q) {
    p->~id<int>();
    q->~id<int>();
    p->id<int>::~id<int>();
    q->id<int>::~id<int>();
    p->template id<int>::~id<int>(); // expected-error {{expected unqualified-id}}
    q->template id<int>::~id<int>(); // expected-error {{expected unqualified-id}}
    p->A::template id<int>::~id<int>();
    q->A::template id<int>::~id<int>();
  }
#endif
}

namespace dr421 { // dr421: yes
  struct X { X(); int n; int &r; };
  int *p = &X().n; // expected-error {{taking the address of a temporary}}
  int *q = &X().r;
}

namespace dr422 { // dr422: yes
  template<typename T, typename U> void f() {
    typedef T type; // expected-note {{prev}}
    typedef U type; // expected-error {{redef}}
  }
  template void f<int, int>();
  template void f<int, char>(); // expected-note {{instantiation}}
}

namespace dr423 { // dr423: yes
  template<typename T> struct X { operator T&(); };
  void f(X<int> x) { x += 1; }
}

namespace dr424 { // dr424: yes
  struct A {
    typedef int N; // expected-note {{previous}}
    typedef int N; // expected-error {{redefinition}}

    struct X;
    typedef X X; // expected-note {{previous}}
    struct X {};

    struct X *p;
    struct A::X *q;
    X *r;

    typedef X X; // expected-error {{redefinition}}
  };
  struct B {
    typedef int N;
  };
  struct C : B {
    typedef int N; // expected-note {{previous}}
    typedef int N; // expected-error {{redefinition}}
  };
}

namespace dr425 { // dr425: yes
  struct A { template<typename T> operator T() const; } a;
  float f = 1.0f * a; // expected-error {{ambiguous}} expected-note 5+{{built-in candidate}}

  template<typename T> struct is_float;
  template<> struct is_float<float> { typedef void type; };

  struct B {
    template<typename T, typename U = typename is_float<T>::type> operator T() const; // expected-error 0-1{{extension}}
  } b;
  float g = 1.0f * b; // ok
}

namespace dr427 { // dr427: yes
  struct B {};
  struct D : public B {
    D(B &) = delete; // expected-error 0-1{{extension}} expected-note {{deleted}}
  };

  extern D d1;
  B &b = d1;
  const D &d2 = static_cast<const D&>(b);
  const D &d3 = (const D&)b;
  const D &d4(b); // expected-error {{deleted}}
}

namespace dr428 { // dr428: yes
  template<typename T> T make();
  extern struct X x; // expected-note 5{{forward declaration}}
  void f() {
    throw void(); // expected-error {{cannot throw}}
    throw make<void*>();
    throw make<const volatile void*>();
    throw x; // expected-error {{cannot throw}}
    throw make<X&>(); // expected-error {{cannot throw}}
    throw make<X*>(); // expected-error {{cannot throw}}
    throw make<const volatile X&>(); // expected-error {{cannot throw}}
    throw make<const volatile X*>(); // expected-error {{cannot throw}}
  }
}

namespace dr429 { // dr429: yes c++11
  // FIXME: This rule is obviously intended to apply to C++98 as well.
  typedef __SIZE_TYPE__ size_t;
  struct A {
    static void *operator new(size_t, size_t);
    static void operator delete(void*, size_t);
  } *a = new (0) A;
#if __cplusplus >= 201103L
  // expected-error@-2 {{'new' expression with placement arguments refers to non-placement 'operator delete'}}
  // expected-note@-4 {{here}}
#endif
  struct B {
    static void *operator new(size_t, size_t);
    static void operator delete(void*);
    static void operator delete(void*, size_t);
  } *b = new (0) B; // ok, second delete is not a non-placement deallocation function
}

namespace dr430 { // dr430: yes c++11
  // resolved by n2239
  // FIXME: This should apply in C++98 too.
  void f(int n) {
    int a[] = { n++, n++, n++ };
#if __cplusplus < 201103L
    // expected-warning@-2 {{multiple unsequenced modifications to 'n'}}
#endif
  }
}

namespace dr431 { // dr431: yes
  struct A {
    template<typename T> T *get();
    template<typename T> struct B {
      template<typename U> U *get();
    };
  };

  template<typename T> void f(A a) {
    a.get<A>()->get<T>();
    a.get<T>()
        ->get<T>(); // expected-error {{use 'template'}}
    a.get<T>()->template get<T>();
    a.A::get<T>();
    A::B<int> *b = a.get<A::B<int> >();
    b->get<int>();
    b->A::B<int>::get<int>();
    b->A::B<int>::get<T>();
    b->A::B<T>::get<int>(); // expected-error {{use 'template'}}
    b->A::B<T>::template get<int>();
    b->A::B<T>::get<T>(); // expected-error {{use 'template'}}
    b->A::B<T>::template get<T>();
    A::B<T> *c = a.get<A::B<T> >();
    c->get<int>(); // expected-error {{use 'template'}}
    c->template get<int>();
  }
}

namespace dr432 { // dr432: yes
  template<typename T> struct A {};
  template<typename T> struct B : A<B> {}; // expected-error {{requires template arguments}} expected-note {{declared}}
  template<typename T> struct C : A<C<T> > {};
#if __cplusplus >= 201103L
  template<typename T> struct D : decltype(A<D>()) {}; // expected-error {{requires template arguments}} expected-note {{declared}}
#endif
}

namespace dr433 { // dr433: yes
  template<class T> struct S {
    void f(union U*);
  };
  U *p;
  template<class T> void S<T>::f(union U*) {}

  S<int> s;
}

namespace dr434 { // dr434: yes
  void f() {
    const int ci = 0;
    int *pi = 0;
    const int *&rpci = pi; // expected-error {{cannot bind}}
    rpci = &ci;
    *pi = 1;
  }
}

// dr435: na

namespace dr436 { // dr436: yes
  enum E { f }; // expected-note {{previous}}
  void f(); // expected-error {{redefinition}}
}

namespace dr437 { // dr437: no
  // This is superseded by 1308, which is in turn superseded by 1330,
  // which restores this rule.
  template<typename U> struct T : U {}; // expected-error {{incomplete}}
  struct S { // expected-note {{not complete}}
    void f() throw(S);
    void g() throw(T<S>); // expected-note {{in instantiation of}}
    struct U; // expected-note {{forward}}
    void h() throw(U); // expected-error {{incomplete}}
    struct U {};
  };
}

// dr438 FIXME write a codegen test
// dr439 FIXME write a codegen test
// dr441 FIXME write a codegen test
// dr442: sup 348
// dr443: na

namespace dr444 { // dr444: yes
  struct D;
  struct B { // expected-note {{candidate is the implicit copy}} expected-note 0-1 {{implicit move}}
    D &operator=(D &) = delete; // expected-error 0-1{{extension}} expected-note {{deleted}}
  };
  struct D : B { // expected-note {{candidate is the implicit}} expected-note 0-1 {{implicit move}}
    using B::operator=;
  } extern d;
  void f() {
    d = d; // expected-error {{deleted}}
  }
}

namespace dr445 { // dr445: yes
  class A { void f(); }; // expected-note {{private}}
  struct B {
    friend void A::f(); // expected-error {{private}}
  };
}

namespace dr446 { // dr446: yes
  struct C;
  struct A {
    A();
    A(const A&) = delete; // expected-error 0-1{{extension}} expected-note +{{deleted}}
    A(const C&);
  };
  struct C : A {};
  void f(A a, bool b, C c) {
    void(b ? a : a);
    b ? A() : a; // expected-error {{deleted}}
    b ? a : A(); // expected-error {{deleted}}
    b ? A() : A(); // expected-error {{deleted}}

    void(b ? a : c);
    b ? a : C(); // expected-error {{deleted}}
    b ? c : A(); // expected-error {{deleted}}
    b ? A() : C(); // expected-error {{deleted}}
  }
}

namespace dr447 { // dr447: yes
  struct A { int n; int a[4]; };
  template<int> struct U {
    typedef int type;
    template<typename V> static void h();
  };
  template<typename T> U<sizeof(T)> g(T);
  template<typename T, int N> void f(int n) {
    // ok, not type dependent
    g(__builtin_offsetof(A, n)).h<int>();
    g(__builtin_offsetof(T, n)).h<int>();
    // value dependent if first argument is a dependent type
    U<__builtin_offsetof(A, n)>::type a;
    U<__builtin_offsetof(T, n)>::type b; // expected-error +{{}} expected-warning 0+{{}}
    // as an extension, we allow the member-designator to include array indices
    g(__builtin_offsetof(A, a[0])).h<int>(); // expected-error {{extension}}
    g(__builtin_offsetof(A, a[N])).h<int>(); // expected-error {{extension}}
    U<__builtin_offsetof(A, a[0])>::type c; // expected-error {{extension}}
    U<__builtin_offsetof(A, a[N])>::type d; // expected-error {{extension}} expected-error +{{}} expected-warning 0+{{}}
  }
}

namespace dr448 { // dr448: yes
  template<typename T = int> void f(int); // expected-error 0-1{{extension}} expected-note {{no known conversion}}
  template<typename T> void g(T t) {
    f<T>(t); // expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
    dr448::f(t); // expected-error {{no matching function}}
  }
  template<typename T> void f(T); // expected-note {{should be declared prior to the call site}}
  namespace HideFromADL { struct X {}; }
  template void g(int); // ok
  template void g(HideFromADL::X); // expected-note {{instantiation of}}
}

// dr449: na

namespace dr450 { // dr450: yes
  typedef int A[3];
  void f1(const A &);
  void f2(A &); // expected-note +{{not viable}}
  struct S { A n; };
  void g() {
    f1(S().n);
    f2(S().n); // expected-error {{no match}}}
  }
#if __cplusplus >= 201103L
  void h() {
    f1(A{});
    f2(A{}); // expected-error {{no match}}
  }
#endif
}

namespace dr482 { // dr482: 3.5
  extern int a;
  void f();

  int dr482::a = 0; // expected-warning {{extra qualification}}
  void dr482::f() {} // expected-warning {{extra qualification}}

  inline namespace X { // expected-error 0-1{{C++11 feature}}
    extern int b;
    void g();
    struct S;
  }
  int dr482::b = 0; // expected-warning {{extra qualification}}
  void dr482::g() {} // expected-warning {{extra qualification}}
  struct dr482::S {}; // expected-warning {{extra qualification}}

  void dr482::f(); // expected-warning {{extra qualification}}
  void dr482::g(); // expected-warning {{extra qualification}}

  // FIXME: The following are valid in DR482's wording, but these are bugs in
  // the wording which we deliberately don't implement.
  namespace N { typedef int type; }
  typedef int N::type; // expected-error {{typedef declarator cannot be qualified}}
  struct A {
    struct B;
    struct A::B {}; // expected-error {{extra qualification}}

#if __cplusplus >= 201103L
    enum class C;
    enum class A::C {}; // expected-error {{extra qualification}}
#endif
  };
}
