// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++11 -verify -triple x86_64-apple-darwin %s

enum E {};

struct Z {}; // expected-note {{here}}
typedef int Integer;

struct X {
  enum E : 1;
  enum E : Z; // expected-error{{invalid underlying type}}
  enum E2 : int;
  enum E3 : Integer;
};

struct Y {
  enum E : int(2);
  enum E : Z(); // expected-error{{not an integer constant}} expected-note {{non-constexpr constructor 'Z'}}
};
