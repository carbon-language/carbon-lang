// RUN: clang-cc -fsyntax-only -verify %s
// PR4794

template <class T> class X
{
  friend class Y;
};
X<int> y;

