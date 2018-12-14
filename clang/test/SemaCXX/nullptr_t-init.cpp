// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -ffreestanding -Wuninitialized %s
// expected-no-diagnostics
typedef decltype(nullptr) nullptr_t;

// Ensure no 'uninitialized when used here' warnings (Wuninitialized), for 
// nullptr_t always-initialized extension.
nullptr_t default_init() {
  nullptr_t a;
  return a;
}
