// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-compatibility

// PR15845
int foo(xxx); // expected-error{{unknown type name}}
