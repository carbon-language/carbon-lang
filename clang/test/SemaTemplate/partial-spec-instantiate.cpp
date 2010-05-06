// RUN: %clang_cc1 -fsyntax-only %s

// PR4607
template <class T> struct X {};

template <> struct X<char>
{
  static char* g();
};

template <class T> struct X2 {};

template <class U>
struct X2<U*> {
  static void f() {
    X<U>::g();
  }
};

void a(char *a, char *b) {X2<char*>::f();}

namespace WonkyAccess {
  template<typename T>
  struct X {
    int m;
  };

  template<typename U>
  class Y;

  template<typename U>
  struct Y<U*> : X<U> { };

  template<>
  struct Y<float*> : X<float> { };

  int f(Y<int*> y, Y<float*> y2) {
    return y.m + y2.m;
  }
}
