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

namespace dr224 { // dr224: no
  namespace example1 {
    template <class T> class A {
      typedef int type;
      A::type a;
      A<T>::type b;
      A<T*>::type c; // expected-error {{missing 'typename'}}
      ::dr224::example1::A<T>::type d;

      class B {
        typedef int type;

        A::type a;
        A<T>::type b;
        A<T*>::type c; // expected-error {{missing 'typename'}}
        ::dr224::example1::A<T>::type d;

        B::type e;
        A<T>::B::type f;
        A<T*>::B::type g; // expected-error {{missing 'typename'}}
        typename A<T*>::B::type h;
      };
    };

    template <class T> class A<T*> {
      typedef int type;
      A<T*>::type a;
      A<T>::type b; // expected-error {{missing 'typename'}}
    };

    template <class T1, class T2, int I> struct B {
      typedef int type;
      B<T1, T2, I>::type b1;
      B<T2, T1, I>::type b2; // expected-error {{missing 'typename'}}

      typedef T1 my_T1;
      static const int my_I = I;
      static const int my_I2 = I+0;
      static const int my_I3 = my_I;
      B<my_T1, T2, my_I>::type b3; // FIXME: expected-error {{missing 'typename'}}
      B<my_T1, T2, my_I2>::type b4; // expected-error {{missing 'typename'}}
      B<my_T1, T2, my_I3>::type b5; // FIXME: expected-error {{missing 'typename'}}
    };
  }

  namespace example2 {
    template <int, typename T> struct X { typedef T type; };
    template <class T> class A {
      static const int i = 5;
      X<i, int>::type w; // FIXME: expected-error {{missing 'typename'}}
      X<A::i, char>::type x; // FIXME: expected-error {{missing 'typename'}}
      X<A<T>::i, double>::type y; // FIXME: expected-error {{missing 'typename'}}
      X<A<T*>::i, long>::type z; // expected-error {{missing 'typename'}}
      int f();
    };
    template <class T> int A<T>::f() {
      return i;
    }
  }
}

// dr225: yes
template<typename T> void dr225_f(T t) { dr225_g(t); } // expected-error {{call to function 'dr225_g' that is neither visible in the template definition nor found by argument-dependent lookup}}
void dr225_g(int); // expected-note {{should be declared prior to the call site}}
template void dr225_f(int); // expected-note {{in instantiation of}}

namespace dr226 { // dr226: no
  template<typename T = void> void f() {}
#if __cplusplus < 201103L
  // expected-error@-2 {{extension}}
  // FIXME: This appears to be wrong: default arguments for function templates
  // are listed as a defect (in c++98) not an extension. EDG accepts them in
  // strict c++98 mode.
#endif
  template<typename T> struct S {
    template<typename U = void> void g();
#if __cplusplus < 201103L
  // expected-error@-2 {{extension}}
#endif
    template<typename U> struct X;
    template<typename U> void h();
  };
  template<typename T> template<typename U> void S<T>::g() {}
  template<typename T> template<typename U = void> struct S<T>::X {}; // expected-error {{cannot add a default template arg}}
  template<typename T> template<typename U = void> void S<T>::h() {} // expected-error {{cannot add a default template arg}}

  template<typename> void friend_h();
  struct A {
    // FIXME: This is ill-formed.
    template<typename=void> struct friend_B;
    // FIXME: f, h, and i are ill-formed.
    //  f is ill-formed because it is not a definition.
    //  h and i are ill-formed because they are not the only declarations of the
    //  function in the translation unit.
    template<typename=void> void friend_f();
    template<typename=void> void friend_g() {}
    template<typename=void> void friend_h() {}
    template<typename=void> void friend_i() {}
#if __cplusplus < 201103L
  // expected-error@-5 {{extension}} expected-error@-4 {{extension}}
  // expected-error@-4 {{extension}} expected-error@-3 {{extension}}
#endif
  };
  template<typename> void friend_i();

  template<typename=void, typename X> void foo(X) {}
  template<typename=void, typename X> struct Foo {}; // expected-error {{missing a default argument}} expected-note {{here}}
#if __cplusplus < 201103L
  // expected-error@-3 {{extension}}
#endif

  template<typename=void, typename X, typename, typename Y> int foo(X, Y);
  template<typename, typename X, typename=void, typename Y> int foo(X, Y);
  int x = foo(0, 0);
#if __cplusplus < 201103L
  // expected-error@-4 {{extension}}
  // expected-error@-4 {{extension}}
#endif
}

void dr227(bool b) { // dr227: yes
  if (b)
    int n;
  else
    int n;
}

namespace dr228 { // dr228: yes
  template <class T> struct X {
    void f();
  };
  template <class T> struct Y {
    void g(X<T> x) { x.template X<T>::f(); }
  };
}

namespace dr229 { // dr229: yes
  template<typename T> void f();
  template<typename T> void f<T*>() {} // expected-error {{function template partial specialization}}
  template<> void f<int>() {}
}

namespace dr231 { // dr231: yes
  namespace outer {
    namespace inner {
      int i; // expected-note {{here}}
    }
    void f() { using namespace inner; }
    int j = i; // expected-error {{undeclared identifier 'i'; did you mean 'inner::i'?}}
  }
}

// dr234: na
// dr235: na

namespace dr236 { // dr236: yes
  void *p = int();
#if __cplusplus < 201103L
  // expected-warning@-2 {{null pointer}}
#else
  // expected-error@-4 {{cannot initialize}}
#endif
}

namespace dr237 { // dr237: dup 470
  template<typename T> struct A { void f() { T::error; } };
  template<typename T> struct B : A<T> {};
  template struct B<int>; // ok
}

namespace dr239 { // dr239: yes
  namespace NS {
    class T {};
    void f(T);
    float &g(T, int);
  }
  NS::T parm;
  int &g(NS::T, float);
  int main() {
    f(parm);
    float &r = g(parm, 1);
    extern int &g(NS::T, float);
    int &s = g(parm, 1);
  }
}

// dr240: dup 616
