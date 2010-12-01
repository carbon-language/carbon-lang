// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++0x -verify -triple x86_64-apple-darwin %s

enum E {};

struct Z {};
typedef int Integer;

struct X {
  enum E : 1;
  enum E : Z; // expected-error{{invalid underlying type}}
  enum E2 : int;
  enum E3 : Integer;
};

struct Y {
  enum E : int(2);
  enum E : Z(); // expected-error{{not an integer constant}}
};
