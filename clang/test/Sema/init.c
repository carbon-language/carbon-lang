// RUN: clang %s -verify -fsyntax-only

typedef void (* fp)(void);
void foo(void);
fp a[1] = { foo };

