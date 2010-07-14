// RUN: %clang_cc1 %s -fsyntax-only -Wignored-qualifiers -verify

int test1() {
  throw;
}

// PR5071
template<typename T> T f() { }

template<typename T>
void g(T t) {
  return t * 2; // okay
}

template<typename T>
T h() {
  return 17;
}

// Don't warn on cv-qualified class return types, only scalar return types.
namespace ignored_quals {
struct S {};
const S class_c();
const volatile S class_cv();

const int scalar_c(); // expected-warning{{'const' type qualifier on return type has no effect}}
const volatile int scalar_cv(); // expected-warning{{'const volatile' type qualifiers on return type have no effect}}
}
