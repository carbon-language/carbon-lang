// RUN: clang-cc %s -pedantic -verify

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
