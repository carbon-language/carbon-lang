// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -trigraphs %s

int x = 000000080;  // expected-error {{invalid digit}}

int y = 0000\
00080;             // expected-error {{invalid digit}}



float X = 1.17549435e-38F;
float Y = 08.123456;

// PR2252
#if -0x8000000000000000  // should not warn.
#endif


char c[] = {
  'df',   // expected-warning {{multi-character character constant}}
  '\t',
  '\\
t',
  '??!',  // expected-warning {{trigraph converted to '|' character}}
  'abcd'  // expected-warning {{multi-character character constant}}
};


#pragma clang diagnostic ignored "-Wmultichar"

char d = 'df'; // no warning.
char e = 'abcd';  // still warn: expected-warning {{multi-character character constant}}

#pragma clang diagnostic ignored "-Wfour-char-constants"

char f = 'abcd';  // ignored.

// rdar://problem/6974641
float t0[] = {
  1.9e20f,
  1.9e-20f,
  1.9e50f,   // expected-error {{too large}}
  1.9e-50f,  // expected-error {{too small}}
  -1.9e20f,
  -1.9e-20f,
  -1.9e50f,  // expected-error {{too large}}
  -1.9e-50f  // expected-error {{too small}}
};
double t1[] = {
  1.9e50,
  1.9e-50,
  1.9e500,   // expected-error {{too large}}
  1.9e-500,  // expected-error {{too small}}
  -1.9e50,
  -1.9e-50,
  -1.9e500,  // expected-error {{too large}}
  -1.9e-500  // expected-error {{too small}}
};
