// RUN: %clang_cc1 -fsyntax-only -verify %s

class X {};

void test() {
  X x;

  x.int; // expected-error{{expected unqualified-id}}
  x.~int(); // expected-error{{expected a class name}}
  x.operator; // expected-error{{expected a type}}
  x.operator typedef; // expected-error{{expected a type}}
}

void test2() {
  X *x;

  x->int; // expected-error{{expected unqualified-id}}
  x->~int(); // expected-error{{expected a class name}}
  x->operator; // expected-error{{expected a type}}
  x->operator typedef; // expected-error{{expected a type}}
}

// PR6327
namespace test3 {
  template <class A, class B> struct pair {};

  void test0() {
    pair<int, int> z = minmax({}); // expected-error {{expected expression}}
  }

  struct string {
    class iterator {};
  };

  void test1() {
    string s;
    string::iterator i = s.foo(); // expected-error {{no member named 'foo'}}
  }
}
