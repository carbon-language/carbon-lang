// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++-specific tests for integral constant expressions.

const int c = 10;
int ar[c];

struct X0 {
  static const int value = static_cast<int>(4.0);
};

void f() {
  if (const int value = 17) {
    int array[value];
  }
}

int a() {
  const int t=t; // expected-note {{subexpression not valid}}
  switch(1) {
    case t:; // expected-error {{not an integer constant expression}}
  }
}
