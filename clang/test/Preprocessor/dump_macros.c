// RUN: clang-cc -E -dM %s -o - | FileCheck %s -strict-whitespace

// Space at end even without expansion tokens
// CHECK: {{#define A[(]x[)] $}}
#define A(x)

// Space before expansion list.
// CHECK: {{#define B[(]x,y[)] x y$}}
#define B(x,y)x y

// No space in argument list.
// CHECK: #define C(x,y) x y
#define C(x, y) x y

// No paste avoidance.
// CHECK: #define X() ..
#define X() ..

// Simple test.
// CHECK: #define Y .
// CHECK: #define Z X()Y
#define Y .
#define Z X()Y

// gcc prints macros at end of translation unit, so last one wins.
// CHECK: #define foo 2
#define foo 1
#undef foo
#define foo 2

