// RUN: clang %s -verify -fsyntax-only

int test1() {
  typedef int x[test1()];  // vla
  static int y = sizeof(x);  // expected-error {{not constant}}
}

