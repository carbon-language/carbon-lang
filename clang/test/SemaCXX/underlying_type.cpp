// RUN: %clang_cc1 -ffreestanding -fsyntax-only -verify -std=c++0x %s

#include "limits.h"

template<typename T, typename U>
struct is_same_type {
  static const bool value = false;
};
template <typename T>
struct is_same_type<T, T> {
  static const bool value = true;
};

__underlying_type(int) a; // expected-error {{only enumeration types}}
__underlying_type(struct b) c; // expected-error {{only enumeration types}}

enum class f : char;
static_assert(is_same_type<char, __underlying_type(f)>::value,
              "f has the wrong underlying type");

enum g {d = INT_MIN };
static_assert(is_same_type<int, __underlying_type(g)>::value,
              "g has the wrong underlying type");

__underlying_type(f) h;
static_assert(is_same_type<char, decltype(h)>::value,
              "h has the wrong type");
