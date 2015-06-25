// RUN: %clang_cc1 -std=c++14 -fconcepts-ts -x c++ %s -verify
// expected-no-diagnostics

// Test parsing of the optional requires-clause in a template-declaration.

template <typename T> requires true
void foo() { }


template <typename T> requires !0
struct A {
  void foo();
  struct AA;
  enum E : int;
  static int x;

  template <typename> requires true
  void Mfoo();

  template <typename> requires true
  struct M;

  template <typename> requires true
  static int Mx;

  template <typename TT> requires true
  using MQ = M<TT>;
};

template <typename T> requires !0
void A<T>::foo() { }

template <typename T> requires !0
struct A<T>::AA { };

template <typename T> requires !0
enum A<T>::E : int { E0 };

template <typename T> requires !0
int A<T>::x = 0;

template <typename T> requires !0
template <typename> requires true
void A<T>::Mfoo() { }

template <typename T> requires !0
template <typename> requires true
struct A<T>::M { };

template <typename T> requires !0
template <typename> requires true
int A<T>::Mx = 0;


template <typename T> requires true
int x = 0;

template <typename T> requires true
using Q = A<T>;

struct C {
  template <typename> requires true
  void Mfoo();

  template <typename> requires true
  struct M;

  template <typename> requires true
  static int Mx;

  template <typename T> requires true
  using MQ = M<T>;
};

template <typename> requires true
void C::Mfoo() { }

template <typename> requires true
struct C::M { };

template <typename> requires true
int C::Mx = 0;
