// RUN: %clang_cc1 -verify -fsyntax-only -std=c++0x %s

// Make sure we know these are legitimate commas and not typos for ';'.
namespace Commas {
  int a,
  b [[ ]],
  c alignas(double);
}

struct S {};
enum E { e };

auto f() -> struct S {
  return S();
}
auto g() -> enum E {
  return E();
}
