// RUN: clang-cc -fsyntax-only -verify %s

enum Enum { val = 1 };
template <Enum v> struct C {
  typedef C<v> Self;
};
template struct C<val>;
