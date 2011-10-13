// Without PCH
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -include %s %s
// With PCH
// RUN: %clang_cc1 -x c++-header -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -include-pch %t %s

#ifndef PASS1
#define PASS1

struct foo {
  foo() = default;
  void bar() = delete; // expected-note{{deleted here}}
};

#else

foo::foo() { } // expected-error{{definition of explicitly defaulted default constructor}}
foo f;
void fn() {
  f.bar(); // expected-error{{deleted function}}
}

#endif
