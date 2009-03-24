// RUN: clang-cc %s -fsyntax-only -verify -std=c99
// rdar://6095180

#include <assert.h>
struct s { char c[17]; };
extern struct s foo(void);

struct s a, b, c;

int A[sizeof((foo().c)) == 17 ? 1 : -1];
int B[sizeof((a.c)) == 17 ? 1 : -1];


// comma does array/function promotion in c99.
int X[sizeof(0, (foo().c)) == sizeof(char*) ? 1 : -1];
int Y[sizeof(0, (a,b).c) == sizeof(char*) ? 1 : -1];
int Z[sizeof(0, (a=b).c) == sizeof(char*) ? 1 : -1];

