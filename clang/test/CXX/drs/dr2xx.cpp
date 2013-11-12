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
