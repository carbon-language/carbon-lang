// RUN: clang-cc -fsyntax-only -verify %s

template <class T> class A {
  typedef int Member;
};

class B {
  template <class T> friend class A;
  template <class T> friend class Undeclared;
  
  // FIXME: Diagnostic below could be (and was) better.
  template <class T> friend typename A<T>::Member; // expected-error {{classes or functions}}
};
