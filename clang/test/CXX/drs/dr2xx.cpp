// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
#else
#define fold
#endif

namespace dr200 { // dr200: dup 214
  template <class T> T f(int);
  template <class T, class U> T f(U) = delete; // expected-error 0-1{{extension}}

  void g() {
    f<int>(1);
  }
}

// dr201 FIXME: write codegen test

namespace dr202 { // dr202: yes
  template<typename T> T f();
  template<int (*g)()> struct X {
    int arr[fold(g == &f<int>) ? 1 : -1];
  };
  template struct X<f>;
}

// FIXME (export) dr204: no

namespace dr206 { // dr206: yes
  struct S; // expected-note 2{{declaration}}
  template<typename T> struct Q { S s; }; // expected-error {{incomplete}}
  template<typename T> void f() { S s; } // expected-error {{incomplete}}
}

namespace dr207 { // dr207: yes
  class A {
  protected:
    static void f() {}
  };
  class B : A {
  public:
    using A::f;
    void g() {
      A::f();
      f();
    }
  };
}

// dr208 FIXME: write codegen test

namespace dr209 { // dr209: yes
  class A {
    void f(); // expected-note {{here}}
  };
  class B {
    friend void A::f(); // expected-error {{private}}
  };
}

// dr210 FIXME: write codegen test

namespace dr211 { // dr211: yes
  struct A {
    A() try {
      throw 0;
    } catch (...) {
      return; // expected-error {{return in the catch of a function try block of a constructor}}
    }
  };
}

namespace dr213 { // dr213: yes
  template <class T> struct A : T {
    void h(T t) {
      char &r1 = f(t);
      int &r2 = g(t); // expected-error {{undeclared}}
    }
  };
  struct B {
    int &f(B);
    int &g(B); // expected-note {{in dependent base class}}
  };
  char &f(B);

  template void A<B>::h(B); // expected-note {{instantiation}}
}

namespace dr214 { // dr214: yes
  template<typename T, typename U> T checked_cast(U from) { U::error; }
  template<typename T, typename U> T checked_cast(U *from);
  class C {};
  void foo(int *arg) { checked_cast<const C *>(arg); }

  template<typename T> T f(int);
  template<typename T, typename U> T f(U) { T::error; }
  void g() {
    f<int>(1);
  }
}

namespace dr215 { // dr215: yes
  template<typename T> class X {
    friend void T::foo();
    int n;
  };
  struct Y {
    void foo() { (void)+X<Y>().n; }
  };
}

namespace dr216 { // dr216: no
  // FIXME: Should reject this: 'f' has linkage but its type does not,
  // and 'f' is odr-used but not defined in this TU.
  typedef enum { e } *E;
  void f(E);
  void g(E e) { f(e); }

  struct S {
    // FIXME: Should reject this: 'f' has linkage but its type does not,
    // and 'f' is odr-used but not defined in this TU.
    typedef enum { e } *E;
    void f(E);
  };
  void g(S s, S::E e) { s.f(e); }
}

namespace dr217 { // dr217: yes
  template<typename T> struct S {
    void f(int);
  };
  template<typename T> void S<T>::f(int = 0) {} // expected-error {{default arguments cannot be added}}
}

namespace dr218 { // dr218: yes
  namespace A {
    struct S {};
    void f(S);
  }
  namespace B {
    struct S {};
    void f(S);
  }

  struct C {
    int f;
    void test1(A::S as) { f(as); } // expected-error {{called object type 'int'}}
    void test2(A::S as) { void f(); f(as); } // expected-error {{too many arguments}} expected-note {{}}
    void test3(A::S as) { using A::f; f(as); } // ok
    void test4(A::S as) { using B::f; f(as); } // ok
    void test5(A::S as) { int f; f(as); } // expected-error {{called object type 'int'}}
    void test6(A::S as) { struct f {}; (void) f(as); } // expected-error {{no matching conversion}} expected-note +{{}}
  };

  namespace D {
    struct S {};
    struct X { void operator()(S); } f;
  }
  void testD(D::S ds) { f(ds); } // expected-error {{undeclared identifier}}

  namespace E {
    struct S {};
    struct f { f(S); };
  }
  void testE(E::S es) { f(es); } // expected-error {{undeclared identifier}}

  namespace F {
    struct S {
      template<typename T> friend void f(S, T) {}
    };
  }
  void testF(F::S fs) { f(fs, 0); }

  namespace G {
    namespace X {
      int f;
      struct A {};
    }
    namespace Y {
      template<typename T> void f(T);
      struct B {};
    }
    template<typename A, typename B> struct C {};
  }
  void testG(G::C<G::X::A, G::Y::B> gc) { f(gc); }
}

// dr219: na
// dr220: na

namespace dr221 { // dr221: yes
  struct A {
    A &operator=(int&);
    A &operator+=(int&);
    static A &operator=(A&, double&); // expected-error {{cannot be a static member}}
    static A &operator+=(A&, double&); // expected-error {{cannot be a static member}}
    friend A &operator=(A&, char&); // expected-error {{must be a non-static member function}}
    friend A &operator+=(A&, char&);
  };
  A &operator=(A&, float&); // expected-error {{must be a non-static member function}}
  A &operator+=(A&, float&);

  void test(A a, int n, char c, float f) {
    a = n;
    a += n;
    a = c;
    a += c;
    a = f;
    a += f;
  }
}

// dr222 is a mystery -- it lists no changes to the standard, and yet was
// apparently both voted into the WP and acted upon by the editor.

// dr223: na
