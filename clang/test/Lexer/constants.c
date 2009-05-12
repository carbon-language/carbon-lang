// RUN: clang-cc -fsyntax-only -verify -pedantic -trigraphs %s

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
