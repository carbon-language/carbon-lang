// RUN: %clang_cc1 %s -verify -fsyntax-only -triple arm-none-linux
class Foo {
  void bar();
  static void bar2();
  unsigned v;
  static unsigned s;
};

void __attribute__((naked)) Foo::bar() { // expected-note{{attribute is here}}
  asm("mov r2, %0" : : "r"(v)); // expected-error{{'this' pointer references not allowed in naked functions}}
}

void __attribute__((naked)) Foo::bar2() {
  asm("mov r2, %0" : : "r"(s)); // static member reference is OK
}
