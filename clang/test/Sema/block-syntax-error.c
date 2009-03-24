// RUN: clang-cc %s -fsyntax-only -verify -fblocks

void (^noop)(void);

void somefunction() {
  noop = ^noop;	// expected-error {{type name requires a specifier or qualifier}} expected-error {{expected expression}}
}
