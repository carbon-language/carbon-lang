// RUN: clang-cc -E %s -o %t &&
// This should print as ".. ." to avoid turning into ...
// RUN: grep -F 'A: . . .' %t &&
#define y(a) ..a
A: y(.)

// RUN: grep -F 'C: .. .' %t &&
#define DOT .
C: ..DOT


// RUN: grep -F 'D: + + - - + + = = =' %t &&
#define PLUS +
#define EMPTY
#define f(x) =x=
D: +PLUS -EMPTY- PLUS+ f(=)

// RUN: grep -F 'E: L "str"' %t

// Should expand to L "str" not L"str"
#define test(x) L#x
E: test(str)

