// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

#include <stddef.h>

template <typename T, typename U> struct same_type {
  static const bool value = false;
};

template <typename T> struct same_type<T, T> {
  static const bool value = true;
};

int operator "" _int (const char *, size_t);
static_assert(same_type<int, decltype(""_int)>::value, "not the same type!");

int i = ""_int;
int j = L""_int; // expected-error {{no matching literal operator function}}

int operator "" _int (const wchar_t *, size_t);

int k = L""_int;

