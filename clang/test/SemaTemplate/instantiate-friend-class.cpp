// RUN: clang-cc -fsyntax-only -verify %s

template <class T> class X
{
  friend class Y;
};
X<int> y;

