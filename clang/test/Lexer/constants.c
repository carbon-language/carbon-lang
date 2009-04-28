// RUN: clang-cc -fsyntax-only -verify %s

int x = 000000080;  // expected-error {{invalid digit}}

int y = 0000\
00080;             // expected-error {{invalid digit}}



float X = 1.17549435e-38F;
float Y = 08.123456;

// PR2252
#if -0x8000000000000000  // should not warn.
#endif
