// Test this without pch.
// RUN: clang-cc -include %S/attrs.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/attrs.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 
// expected-note{{previous overload}}
double f(double); // expected-error{{overloadable}}
