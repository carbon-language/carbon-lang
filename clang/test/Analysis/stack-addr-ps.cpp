// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

// FIXME: Only the stack-address checking in Sema catches this right now, and
// the stack analyzer doesn't handle the ImplicitCastExpr (lvalue).
const int& g() {
  int s;
  return s; // expected-warning{{reference to stack memory associated with local variable 's' returned}}
}
