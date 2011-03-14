// RUN: %clang_cc1 -std=c++0x -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// This is the "let the user shoot himself in the foot" clause.
void f() noexcept {
  throw 0; // no-error
}
void g() throw() {
  throw 0; // no-error
}
void h() throw(int) {
  throw 0.0; // no-error
}
