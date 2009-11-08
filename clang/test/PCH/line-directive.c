// Test this without pch.
// RUN: clang-cc -include %S/line-directive.h -fsyntax-only %s 2>&1|grep "25:5"

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/line-directive.h
// RUN: clang-cc -include-pch %t -fsyntax-only %s 2>&1|grep "25:5"  

double x; // expected-error{{redefinition of 'x' with a different type}}
















// expected-note{{previous definition is here}}
