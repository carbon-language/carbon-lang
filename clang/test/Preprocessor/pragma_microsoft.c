// RUN: clang-cc %s -fsyntax-only -verify -fms-extensions

// rdar://6495941

#define FOO 1
#define BAR "2"

#pragma comment(linker,"foo=" FOO) // expected-error {{pragma comment requires parenthesized identifier and optional string}}
#pragma comment(linker," bar=" BAR)

#pragma comment( user, "Compiled on " __DATE__ " at " __TIME__ ) 

#pragma comment(foo)    // expected-error {{unknown kind of pragma comment}}
#pragma comment(compiler,)     // expected-error {{pragma comment requires}}
#define foo compiler
#pragma comment(foo)   // macro expand kind.
#pragma comment(foo) x // expected-error {{pragma comment requires}}

#pragma comment(user, "foo\abar\nbaz\tsome	thing")

