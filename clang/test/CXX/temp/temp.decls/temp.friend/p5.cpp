// RUN: clang-cc -fsyntax-only -verify %s

class A {
  template <class T> friend class B;
};

