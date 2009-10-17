// RUN: clang-cc -fsyntax-only -verify %s

void test() {
  bool x = true;
  switch (x) { // expected-warning {{bool}}
    case 0:
      break;
  }

  int n = 3;
  switch (n && 1) { // expected-warning {{bool}}
    case 1:
      break;
  }
}
