// RUN: clang-cc %s -pedantic -verify -triple=x86_64-apple-darwin9

// rdar://6097662
typedef int (*T)[2];
restrict T x;

typedef int *S[2];
restrict S y; // expected-error {{restrict requires a pointer or reference ('S' (aka 'int *[2]') is invalid)}}



// int128_t is available.
int a() {
  __int128_t s;
  __uint128_t t;
}
// but not a keyword
int b() {
  int __int128_t;
  int __uint128_t;
}


// Array type merging should convert array size to whatever matches the target
// pointer size.
// rdar://6880874
extern int i[1LL];
int i[(short)1];

enum e { e_1 };
extern int j[sizeof(enum e)];  // expected-note {{previous definition}}
int j[42];   // expected-error {{redefinition of 'j' with a different type}}

// rdar://6880104
_Decimal32 x;  // expected-error {{GNU decimal type extension not supported}}


// rdar://6880951
int __attribute__ ((vector_size (8), vector_size (8))) v;  // expected-error {{invalid vector type}}
