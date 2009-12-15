//RUN: %clang_cc1 -fsyntax-only -verify %s

char * c;
char const ** c2 = &c; // expected-warning {{initializing, 'char const **' and 'char **' have different qualifiers in nested pointer types}}

typedef char dchar;
dchar *** c3 = &c2; // expected-warning {{initializing, 'dchar ***' and 'char const ***' have different qualifiers in nested pointer types}}

volatile char * c4;
char ** c5 = &c4; // expected-warning {{initializing, 'char **' and 'char volatile **' have different qualifiers in nested pointer types}}
