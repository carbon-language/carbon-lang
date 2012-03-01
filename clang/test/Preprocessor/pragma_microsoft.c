// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions

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


// __pragma

__pragma(comment(linker," bar=" BAR))                                            

#define MACRO_WITH__PRAGMA { \
  __pragma(warning(push)); \
  __pragma(warning(disable: 10000)); \
  2+2; \
  __pragma(warning(pop)); \
}

void f()
{
  __pragma()

  // If we ever actually *support* __pragma(warning(disable: x)),
  // this warning should go away.
  MACRO_WITH__PRAGMA // expected-warning {{expression result unused}}
}
