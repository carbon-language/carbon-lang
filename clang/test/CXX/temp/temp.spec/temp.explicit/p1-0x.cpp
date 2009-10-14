// RUN: clang-cc -fsyntax-only -std=c++0x -verify %s

template<typename T>
struct X {
  void f();
};

template inline void X<int>::f(); // expected-error{{'inline'}}

// FIXME: test constexpr
