// RUN: %clang_cc1 -std=c++11 -verify %s

namespace UseBeforeDefinition {
  struct A {
    template<typename T> static constexpr T get() { return T(); }
    // ok, not a constant expression.
    int n = get<int>();
  };

  // ok, constant expression.
  constexpr int j = A::get<int>();

  template<typename T> constexpr int consume(T);
  // ok, not a constant expression.
  const int k = consume(0); // expected-note {{here}}

  template<typename T> constexpr int consume(T) { return 0; }
  // ok, constant expression.
  constexpr int l = consume(0);

  constexpr int m = k; // expected-error {{constant expression}} expected-note {{initializer of 'k'}}
}

namespace IntegralConst {
  template<typename T> constexpr T f(T n) { return n; }
  enum E {
    v = f(0), w = f(1) // ok
  };
  static_assert(w == 1, "");

  char arr[f('x')]; // ok
  static_assert(sizeof(arr) == 'x', "");
}

namespace ConvertedConst {
  template<typename T> constexpr T f(T n) { return n; }
  int f() {
    switch (f()) {
      case f(4): return 0;
    }
    return 1;
  }
}

namespace OverloadResolution {
  template<typename T> constexpr T f(T t) { return t; }

  template<int n> struct S { };

  template<typename T> auto g(T t) -> S<f(sizeof(T))> &;
  char &f(...);

  template<typename T> auto h(T t[f(sizeof(T))]) -> decltype(&*t) {
    return t;
  }

  S<4> &k = g(0);
  int *p, *q = h(p);
}

namespace DataMember {
  template<typename T> struct S { static const int k; };
  const int n = S<int>::k; // expected-note {{here}}
  template<typename T> const int S<T>::k = 0;
  constexpr int m = S<int>::k; // ok
  constexpr int o = n; // expected-error {{constant expression}} expected-note {{initializer of 'n'}}
}

namespace Reference {
  const int k = 5;
  template<typename T> struct S {
    static volatile int &r;
  };
  template<typename T> volatile int &S<T>::r = const_cast<volatile int&>(k);
  constexpr int n = const_cast<int&>(S<int>::r);
  static_assert(n == 5, "");
}
