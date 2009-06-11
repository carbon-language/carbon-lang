// RUN: clang-cc -fsyntax-only -verify %s

template<int i> struct x {
  static const int j = i;
  x<j>* y;
};

