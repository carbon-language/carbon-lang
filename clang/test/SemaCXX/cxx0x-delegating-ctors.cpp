// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s

struct foo {
  int i;
  foo();
  foo(int);
  foo(int, int);
  foo(bool);
  foo(char);
  foo(const float*);
  foo(const float&);
  foo(void*);
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
foo::foo (const float* f) : foo(*f) {
}

foo::foo (const float &f) : foo(&f) { //expected-error{{delegates to itself}}
}

foo::foo (char) : i(3), foo(3) { // expected-error{{must appear alone}}
}

// This should not cause an infinite loop
foo::foo (void*) : foo(4.0f) {
}

struct deleted_dtor {
  ~deleted_dtor() = delete; // expected-note{{function has been explicitly marked deleted here}}
  deleted_dtor();
  deleted_dtor(int) : deleted_dtor() // expected-error{{attempt to use a deleted function}}
  {}
};
