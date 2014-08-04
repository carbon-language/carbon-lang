// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct S {
  template <typename Ty = char>
  static_assert(sizeof(Ty) != 1, "Not a char"); // expected-error {{a static_assert declaration cannot be a template}}
};

template <typename Ty = char>
static_assert(sizeof(Ty) != 1, "Not a char"); // expected-error {{a static_assert declaration cannot be a template}}
