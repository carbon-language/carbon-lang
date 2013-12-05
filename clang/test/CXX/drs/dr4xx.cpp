// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

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
