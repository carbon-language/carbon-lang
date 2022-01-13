// RUN: %clang_cc1 "-triple" "arm-linux-androideabi16" -fsyntax-only -verify %s

void f0(int) __attribute__((availability(android,introduced=14,deprecated=19)));
void f1(int) __attribute__((availability(android,introduced=16)));
void f2(int) __attribute__((availability(android,introduced=14,deprecated=16))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(android,introduced=19)));
void f4(int) __attribute__((availability(android,introduced=9,deprecated=11,obsoleted=16), availability(ios,introduced=2.0,deprecated=3.0))); // expected-note{{explicitly marked unavailable}}
void f5(int) __attribute__((availability(ios,introduced=3.2), availability(android,unavailable))); // expected-note{{'f5' has been explicitly marked unavailable here}}

void test() {
  f0(0);
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in Android 16}}
  f3(0);
  f4(0); // expected-error{{f4' is unavailable: obsoleted in Android 16}}
  f5(0); // expected-error{{'f5' is unavailable: not available on Android}}
}

// rdar://10535640

enum {
    foo __attribute__((availability(android,introduced=8.0,deprecated=9.0)))
};

enum {
    bar __attribute__((availability(android,introduced=8.0,deprecated=9.0))) = foo
};

enum __attribute__((availability(android,introduced=8.0,deprecated=9.0))) {
    bar1 = foo
};
