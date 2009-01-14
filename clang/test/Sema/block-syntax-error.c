// RUN: clang %s -fsyntax-only -verify -fblocks

void (^noop)(void);

void somefunction() {
  noop = ^int *{}; // expected-error {{expected expression}}

  noop = ^noop;	// expected-error {{expected expression}}
}
