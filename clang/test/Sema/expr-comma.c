// RUN: clang %s -fsyntax-only -verify
// rdar://6095180

#include <assert.h>
struct s { char c[17]; };
extern struct s foo (void);

// sizeof 'c' should be 17, not sizeof(char*).
int X[sizeof(0, (foo().c)) == 17 ? 1 : -1];


