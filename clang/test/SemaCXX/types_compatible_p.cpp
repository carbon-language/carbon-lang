// RUN: clang-cc -fsyntax-only -verify %s

bool f() {
  return __builtin_types_compatible_p(int, const int); // expected-error{{C++}}
}
