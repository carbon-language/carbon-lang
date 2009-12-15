// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo();
void foo() __attribute__((unused));
void foo() __attribute__((unused));
void foo(){} // expected-note {{previous definition is here}}
void foo() __attribute__((constructor)); // expected-warning {{must precede definition}}
void foo();

int bar;
extern int bar;
int bar;
int bar __attribute__((weak));
int bar __attribute__((used));
extern int bar __attribute__((weak));
int bar = 0; // expected-note {{previous definition is here}}
int bar __attribute__((weak)); // expected-warning {{must precede definition}}
int bar;

