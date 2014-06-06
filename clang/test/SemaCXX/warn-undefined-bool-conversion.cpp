// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wundefined-bool-conversion %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-bool-conversion -Wundefined-bool-conversion %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wbool-conversion %s

void test1(int &x) {
  if (x == 1) { }
  if (&x) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

  if (!&x) { }
  // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
}

class test2 {
  test2() : x(y) {}

  void foo() {
    if (this) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; pointer may be assumed to always convert to true}}

    if (!this) { }
    // expected-warning@-1{{'this' pointer cannot be null in well-defined C++ code; pointer may be assumed to always convert to true}}
  }

  void bar() {
    if (x == 1) { }
    if (&x) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}

    if (!&x) { }
    // expected-warning@-1{{reference cannot be bound to dereferenced null pointer in well-defined C++ code; pointer may be assumed to always convert to true}}
  }

  int &x;
  int y;
};
