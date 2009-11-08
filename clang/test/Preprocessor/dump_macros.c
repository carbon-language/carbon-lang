// RUN: clang-cc -E -dM %s -o %t

// Space even without expansion tokens
// RUN: grep "#define A(x) " %t
#define A(x)

// Space before expansion list.
// RUN: grep "#define B(x,y) x y" %t
#define B(x,y)x y

// No space in expansion list.
// RUN: grep "#define C(x,y) x y" %t
#define C(x, y) x y

// No paste avoidance.
// RUN: grep "#define X() .." %t
#define X() ..

// Simple test.
// RUN: grep "#define Y ." %t
// RUN: grep "#define Z X()Y" %t
#define Y .
#define Z X()Y

// gcc prints macros at end of translation unit, so last one wins.
// RUN: grep "#define foo 2" %t
// RUN: not grep "#define foo 1" %t
#define foo 1
#undef foo
#define foo 2

