// RUN: %clang_cc1 -verify %s

// These must be the last lines in this test.
// expected-error@+1{{requires a parenthesized string}} expected-error@+1{{expected}}
int i = __has_warning();
