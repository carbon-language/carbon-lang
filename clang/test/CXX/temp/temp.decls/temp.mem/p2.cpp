// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename>
void quux();

void fun() {
  struct foo {
    template <typename> struct bar {};  // expected-error{{templates cannot be declared inside of a local class}}
    template <typename> void baz() {}   // expected-error{{templates cannot be declared inside of a local class}}
    template <typename> void qux();     // expected-error{{templates cannot be declared inside of a local class}}
  };
}
