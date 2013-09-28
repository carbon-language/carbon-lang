// RUN: %clang_cc1 -std=c++1y %s -verify

// expected-no-diagnostics
namespace variadic_expansion {
  void f(int &, char &);

  template <typename ... T> void g(T &... t) {
    f([&a(t)]()->decltype(auto) {
      return a;
    }() ...);
  }

  void h(int i, char c) { g(i, c); }
}
