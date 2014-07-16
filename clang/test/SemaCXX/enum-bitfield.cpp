// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++11 -verify -triple x86_64-apple-darwin %s

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
  enum E : Z(); // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'Z'}}
};

namespace pr18587 {
struct A {
  enum class B {
    C
  };
};
const int C = 4;
struct D {
  A::B : C;
};
}
