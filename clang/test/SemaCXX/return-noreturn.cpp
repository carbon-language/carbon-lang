// RUN: %clang_cc1 %s -fsyntax-only -verify -Wreturn-type -Wno-unreachable-code

// <rdar://problem/8875247> - Properly handle CFGs with destructors.
struct rdar8875247 {
  ~rdar8875247 ();
};
void rdar8875247_aux();

int rdar8875247_test() {
  rdar8875247 f;
} // expected-warning{{control reaches end of non-void function}}
