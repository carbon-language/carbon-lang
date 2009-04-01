// RUN: clang -fsyntax-only -pedantic %s 

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. Eventually,
   we would like to actually try to perform the suggested
   modifications and compile the result to test that no warnings
   remain. */

void f0(void) { };

struct s {
  int x, y;;
};

_Complex cd;

struct s s0 = { y: 5 };
int array0[5] = { [3] 3 };
