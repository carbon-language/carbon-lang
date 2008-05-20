// RUN: clang -fsyntax-only %s -verify

x(a) int a; {return a;}
y(b) int b; {return a;} // expected-error {{use of undeclared identifier}}

