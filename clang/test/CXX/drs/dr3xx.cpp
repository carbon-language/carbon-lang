// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr300 { // dr300: yes
  template<typename R, typename A> void f(R (&)(A)) {}
  int g(int);
  void h() { f(g); }
}

namespace dr301 { // dr301 WIP
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
}
