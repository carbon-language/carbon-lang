// RUN: clang %s -verify -fsyntax-only

struct simple { int i; };

void f(void) {
   struct simple s[1];
   s->i = 1;
}

