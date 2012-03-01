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


// This should include macro_arg_directive even though the include
// is looking for test.h  This allows us to assign to "n"
#pragma include_alias("test.h", "macro_arg_directive.h" )
#include "test.h"
void test( void ) {
  n = 12;
}

#pragma include_alias("foo.h", <bar.h>) // expected-warning {{pragma include_alias requires matching include directives (e.g include_alias("foo.h", "bar.h") or include_alias(<foo.h>, <bar.h>))}}
#pragma include_alias("test.h") // expected-warning {{pragma include_alias expected ','}}

// Make sure that the names match exactly for a replacement, including path information.  If
// this were to fail, we would get a file not found error
#pragma include_alias(".\pp-record.h", "does_not_exist.h")
#include "pp-record.h"
