// RUN: %clang_cc1 -fsyntax-only -verify %s
void f0() __attribute__((availability(macosx,introduced=10.2,deprecated=10.4,obsoleted=10.6)));

void f1() __attribute__((availability(macosx,deprecated=10.4,introduced=10.2,obsoleted=10.6)));

void f2() __attribute__((availability(ios,deprecated=10.4.7,introduced=10,obsoleted=10.6)));

void f3() __attribute__((availability(ios,deprecated=10.4.7,introduced=10,obsoleted=10.6,introduced=10.2))); // expected-error{{redundant 'introduced' availability change; only the last specified change will be used}}

