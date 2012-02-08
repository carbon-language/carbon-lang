// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

// Check that analysis-based warnings work in lambda bodies.
void analysis_based_warnings() {
  []() -> int { }; // expected-warning{{control reaches end of non-void function}} \
  // expected-error{{lambda expressions are not supported yet}}
}

// FIXME: Also check translation of captured vars to data members,
// most of which isn't in the AST.


