// RUN: clang-cc -fsyntax-only -verify %s

template <class T> class A {
  class Member {
  };
};

class B {
  template <class T> friend class A<T>::Member;
};

A<int> a;
B b;
