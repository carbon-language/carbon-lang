// RUN: %clang_cc1 -std=c++1z %s -verify

namespace hana_enable_if_idiom {
  template<bool> struct A {};
  template<typename, typename = A<true>> struct B;
  template<typename T, bool N> struct B<T, A<N>> {};
  template<typename T> struct B<T, A<T::value>> {};
  struct C {
    static const bool value = true;
  };
  B<C> b;
}

// Ensure that we implement the check that deduced A == A during function
// template partial ordering.
namespace check_substituted_type_matches {
  struct X { typedef int type; };

  // A specific but dependent type is neither better nor worse than a
  // specific and non-dependent type.
  template<typename T> void f(T, typename T::type); // expected-note {{candidate}}
  template<typename T> void f(T, int); // expected-note {{candidate}}
  void test_f() { f(X{}, 0); } // expected-error {{ambiguous}}

  // A specific but dependent type is more specialized than a
  // deducible type.
  template<typename T> void g(T, typename T::type);
  template<typename T, typename U> void g(T, U);
  void test_g() { g(X{}, 0); }
}
