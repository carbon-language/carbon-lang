// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-gcc-compat

namespace N {
  void c1(int *a) {}
}

void t1() {
  int v1 __attribute__((cleanup(N::c1)));
  int v2 __attribute__((cleanup(N::c2)));  // expected-error {{no member named 'c2' in namespace 'N'}}
}
