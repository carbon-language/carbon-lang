// Test this without pch.
// RUN: %clang_cc1 -include %S/attrs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/attrs.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 
// expected-note{{previous overload}}
double f(double); // expected-error{{overloadable}}
