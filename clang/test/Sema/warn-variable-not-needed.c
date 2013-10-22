// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s
// expected-no-diagnostics

static int a;
int bar() {
  extern int a;
  return a;
}
static int a;
