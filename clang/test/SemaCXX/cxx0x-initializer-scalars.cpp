// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace integral {

  void initialization() {
    { const int a{}; static_assert(a == 0, ""); }
    { const int a = {}; static_assert(a == 0, ""); }
    { const int a{1}; static_assert(a == 1, ""); }
    { const int a = {1}; static_assert(a == 1, ""); }
    { const int a{1, 2}; } // expected-error {{excess elements}}
    { const int a = {1, 2}; } // expected-error {{excess elements}}
    // FIXME: Redundant warnings.
    { const short a{100000}; } // expected-error {{cannot be narrowed}} expected-note {{inserting an explicit cast}} expected-warning {{changes value}}
    { const short a = {100000}; } // expected-error {{cannot be narrowed}} expected-note {{inserting an explicit cast}} expected-warning {{changes value}}
  }

  int direct_usage() {
    int ar[10];
    (void) ar[{1}]; // expected-error {{array subscript is not an integer}}

    return {1};
  }

  void inline_init() {
    (void) int{1};
    (void) new int{1};
  }

  struct A {
    int i;
    A() : i{1} {}
  };

}
