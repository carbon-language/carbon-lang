// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR4794

template <class T> class X
{
  friend class Y;
};
X<int> y;

