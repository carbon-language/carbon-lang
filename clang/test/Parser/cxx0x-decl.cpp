// RUN: %clang_cc1 -verify -fsyntax-only -std=c++11 -pedantic %s

// Make sure we know these are legitimate commas and not typos for ';'.
namespace Commas {
  int a,
  b [[ ]],
  c alignas(double);
}

struct S {};
enum E { e, };

auto f() -> struct S {
  return S();
}
auto g() -> enum E {
  return E();
}

class ExtraSemiAfterMemFn {
  // Due to a peculiarity in the C++11 grammar, a deleted or defaulted function
  // is permitted to be followed by either one or two semicolons.
  void f() = delete // expected-error {{expected ';' after delete}}
  void g() = delete; // ok
  void h() = delete;; // ok
  void i() = delete;;; // expected-warning {{extra ';' after member function definition}}
};

// This is technically okay, but not likely what the user expects, so we will
// pedantically warn on it
int *const const p = 0; // expected-warning {{duplicate 'const' declaration specifier}}
const const int *q = 0; // expected-warning {{duplicate 'const' declaration specifier}}
