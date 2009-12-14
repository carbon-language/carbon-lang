// RUN: clang -cc1 -fsyntax-only -verify %s

// FIXME: This is a horrible error message here. Fix.
int @"s" = 5;  // expected-error {{prefix attribute must be}}


// rdar://6480479
@interface A
}; // expected-error {{missing @end}} expected-error {{expected external declaration}}

