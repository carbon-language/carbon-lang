// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename T, T ...Values> struct value_tuple {};

template<typename T>
struct X0 {
  template<T ...Values>
  void f(value_tuple<T, Values...> * = 0);
};

void test_X0() {
  X0<int>().f<1, 2, 3, 4, 5>();
}
