// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {
  operator int&();
};

struct B {
  operator long&();
};

struct C : B, A { };

void test(C c) {
  ++c; // expected-error {{use of overloaded operator '++' is ambiguous}}\
       // expected-note 4 {{built-in candidate operator ++ (}}
}


