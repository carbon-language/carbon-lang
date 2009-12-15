// RUN: %clang_cc1 -emit-llvm-only -verify %s

// This lame little test was ripped straight from the standard.

namespace {
  int i; // expected-note {{candidate}}
}
void test0() { i++; }

namespace A {
  namespace {
    int i; // expected-note {{candidate}}
    int j;
  }
  void test1() { i++; }
}

using namespace A;

void test2() {
  i++; // expected-error {{reference to 'i' is ambiguous}}
  A::i++;
  j++;
}
