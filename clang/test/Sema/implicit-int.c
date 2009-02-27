// RUN: clang -fsyntax-only %s -verify -pedantic

foo() { // expected-warning {{type specifier missing, defaults to 'int'}}
}

y;  // expected-warning {{type specifier missing, defaults to 'int'}}

// rdar://6131634
void f((x));  // expected-warning {{type specifier missing, defaults to 'int'}}

