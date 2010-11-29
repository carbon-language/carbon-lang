// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

// FIXME: Only the stack-address checking in Sema catches this right now, and
// the stack analyzer doesn't handle the ImplicitCastExpr (lvalue).
const int& g() {
  int s;
  return s; // expected-warning{{reference to stack memory associated with local variable 's' returned}}
}

int get_value();

const int &get_reference1() { return get_value(); } // expected-warning {{returning reference to local temporary}}

const int &get_reference2() {
  int const& w2 = get_value(); // expected-note {{binding variable 'w2' to temporary here}}
  return w2; // expected-warning {{reference to temporary associated with local variable 'w2' returned}}
}
