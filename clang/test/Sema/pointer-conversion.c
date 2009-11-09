//RUN: clang-cc -fsyntax-only -verify %s

char * c;
char const ** c2 = &c; // expected-error {{initializing, 'char const **' and 'char **' have different qualifiers in nested pointer types}}

typedef char dchar;
dchar *** c3 = &c2; // expected-error {{initializing, 'dchar ***' and 'char const ***' have different qualifiers in nested pointer types}}

volatile char * c4;
char ** c5 = &c4; // expected-error {{initializing, 'char **' and 'char volatile **' have different qualifiers in nested pointer types}}
