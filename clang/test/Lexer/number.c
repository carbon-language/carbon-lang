// RUN: clang %s -fsyntax-only

float X = 1.17549435e-38F;
float Y = 08.123456;

// PR2252
#if -0x8000000000000000  // should not warn.
#endif

