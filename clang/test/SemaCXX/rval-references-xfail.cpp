// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s
// XFAIL: *
struct MoveOnly {
  MoveOnly();
  MoveOnly(const MoveOnly&) = delete;	// expected-note {{candidate function}} \
  // expected-note 3{{explicitly marked deleted here}}
  MoveOnly(MoveOnly&&);	// expected-note {{candidate function}}
  MoveOnly(int&&);	// expected-note {{candidate function}}
};

MoveOnly returning() {
  MoveOnly mo;
  return mo;
}
