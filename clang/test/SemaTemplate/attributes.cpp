// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int N>
struct X {
  struct __attribute__((__aligned__((N)))) Aligned { }; // expected-error{{'aligned' attribute requires integer constant}}

  int __attribute__((__address_space__(N))) *ptr; // expected-error{{attribute requires 1 argument(s)}}
};
