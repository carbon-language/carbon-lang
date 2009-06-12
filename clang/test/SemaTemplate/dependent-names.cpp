// RUN: clang-cc -fsyntax-only -verify %s 

typedef double A;
template<typename T> class B {
  typedef int A;
};

template<typename T> struct X : B<T> {
  static A a;
};

int a0[sizeof(X<int>::a) == sizeof(double) ? 1 : -1];

// PR4365.
template<class T> class Q;
template<class T> class R : Q<T> {T current;};
