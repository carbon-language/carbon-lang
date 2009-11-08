// Test this without pch.
// RUN: clang-cc -include %S/nonvisible-external-defs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/nonvisible-external-defs.h
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

int g(int, float); // expected-error{{conflicting types}}

// expected-note{{previous declaration}}
