// RUN: %clang_cc1 -fsyntax-only -verify %s

void f0() __attribute__((availability(macosx,introduced=10.4,deprecated=10.2))); // expected-warning{{feature cannot be deprecated in Mac OS X version 10.2 before it was introduced in version 10.4; attribute ignored}}
void f1() __attribute__((availability(ios,obsoleted=2.1,deprecated=3.0)));  // expected-warning{{feature cannot be obsoleted in iOS version 2.1 before it was deprecated in version 3.0; attribute ignored}}

void f2() __attribute__((availability(otheros,introduced=2.2))); // expected-warning{{unknown platform 'otheros' in availability macro}}
