// RUN: clang %s -pedantic -verify

// rdar://6097662
typedef int (*T)[2];
restrict T x;

typedef int *S[2];
restrict S y; // expected-error {{restrict requires a pointer or reference ('S' is invalid)}}


