// RUN: clang %s -verify -fsyntax-only
typedef char T[4];

T foo(int n, int m) {  }  // expected-error {{cannot return array or function}}

void foof(const char *, ...) __attribute__((__format__(__printf__, 1, 2))), barf (void);


