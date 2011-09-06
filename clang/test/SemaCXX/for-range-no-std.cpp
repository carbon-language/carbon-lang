// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++0x-extensions

struct S {
  int *begin();
  int *end();
};

struct T {
};

struct Range {};
int begin(Range); // expected-note {{not viable}}
int end(Range);

namespace NS {
  struct ADL {};
  struct iter {
    int operator*();
    bool operator!=(iter);
    void operator++();
  };
  iter begin(ADL); // expected-note {{not viable}}
  iter end(ADL);

  struct NoADL {};
}
NS::iter begin(NS::NoADL); // expected-note {{not viable}}
NS::iter end(NS::NoADL);

void f() {
  int a[] = {1, 2, 3};
  for (auto b : S()) {} // ok
  for (auto b : T()) {} // expected-error {{no matching function for call to 'begin'}} expected-note {{range has type}}
  for (auto b : a) {} // ok
  for (int b : NS::ADL()) {} // ok
  for (int b : NS::NoADL()) {} // expected-error {{no matching function for call to 'begin'}} expected-note {{range has type}}
}
