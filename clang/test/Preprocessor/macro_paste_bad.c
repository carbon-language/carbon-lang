// RUN: clang-cc -Eonly -verify -pedantic %s
// pasting ""x"" and ""+"" does not give a valid preprocessing token
#define XYZ  x ## +   // expected-error {{pasting formed 'x', an invalid preprocessing token}}
XYZ

// GCC PR 20077
// RUN: clang-cc -Eonly %s -verify

#define a   a ## ## // expected-error {{'##' cannot appear at end of macro expansion}}
#define b() b ## ## // expected-error {{'##' cannot appear at end of macro expansion}}
#define c   c ##    // expected-error {{'##' cannot appear at end of macro expansion}}
#define d() d ##    // expected-error {{'##' cannot appear at end of macro expansion}}


#define e   ## ## e // expected-error {{'##' cannot appear at start of macro expansion}}
#define f() ## ## f // expected-error {{'##' cannot appear at start of macro expansion}}
#define g   ## g    // expected-error {{'##' cannot appear at start of macro expansion}}
#define h() ## h    // expected-error {{'##' cannot appear at start of macro expansion}}
#define i   ##      // expected-error {{'##' cannot appear at start of macro expansion}}
#define j() ##      // expected-error {{'##' cannot appear at start of macro expansion}}


