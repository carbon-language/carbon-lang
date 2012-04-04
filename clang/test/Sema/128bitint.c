// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin9 %s
typedef int i128 __attribute__((__mode__(TI)));
typedef unsigned u128 __attribute__((__mode__(TI)));

int a[((i128)-1 ^ (i128)-2) == 1 ? 1 : -1];
int a[(u128)-1 > 1LL ? 1 : -1];

// PR5435
__uint128_t b = (__uint128_t)-1;

// PR11916: Support for libstdc++ 4.7
__int128 i = (__int128)0;
unsigned __int128 u = (unsigned __int128)-1;
