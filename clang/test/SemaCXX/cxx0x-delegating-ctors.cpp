// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s

struct foo {
  int i;
  foo();
  foo(int);
  foo(int, int);
  foo(bool);
  foo(char);
  foo(float*);
  foo(float&);
};

// Good
foo::foo (int i) : i(i) {
}
// Good
foo::foo () : foo(-1) {
}
// Good
foo::foo (int, int) : foo() {
}

foo::foo (bool) : foo(true) { // expected-error{{delegates to itself}}
}

// Good
foo::foo (float* f) : foo(*f) {
}

// FIXME: This should error
foo::foo (float &f) : foo(&f) {
}

foo::foo (char) : i(3), foo(3) { // expected-error{{must appear alone}}
}
