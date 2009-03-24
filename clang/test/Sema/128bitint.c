// RUN: clang-cc -fsyntax-only -verify %s
typedef int i128 __attribute__((__mode__(TI)));
typedef unsigned u128 __attribute__((__mode__(TI)));

int a[((i128)-1 ^ (i128)-2) == 1 ? 1 : -1];
int a[(u128)-1 > 1LL ? 1 : -1];
