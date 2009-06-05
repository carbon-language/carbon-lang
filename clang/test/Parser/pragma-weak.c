// RUN: clang-cc -fsyntax-only -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

int x;
/* expected-warning {{expected identifier in '#pragma weak'}}*/ #pragma weak
#pragma weak x
#pragma weak y
int y;

/* expected-warning {{expected identifier in '#pragma weak'}}*/ #pragma weak z = =
/* expected-warning {{expected identifier in '#pragma weak'}}*/ #pragma weak z =
#pragma weak z = y

/* expected-warning {{extra tokens at end of '#pragma weak'}}*/ #pragma weak a b
/* expected-warning {{extra tokens at end of '#pragma weak'}}*/ #pragma weak a = x c
