// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions
// REQUIRES: non-ps4-sdk

// rdar://6495941

#define FOO 1
#define BAR "2"

#pragma comment(linker,"foo=" FOO) // expected-error {{pragma comment requires parenthesized identifier and optional string}}
#pragma comment(linker," bar=" BAR)

#pragma comment( user, "Compiled on " __DATE__ " at " __TIME__ ) 

#pragma comment(foo)    // expected-error {{unknown kind of pragma comment}}
#pragma comment(compiler,)     // expected-error {{expected string literal in pragma comment}}
#define foo compiler
#pragma comment(foo)   // macro expand kind.
#pragma comment(foo) x // expected-error {{pragma comment requires}}

#pragma comment(user, "foo\abar\nbaz\tsome	thing")

#pragma detect_mismatch("test", "1")
#pragma detect_mismatch()  // expected-error {{expected string literal in pragma detect_mismatch}}
#pragma detect_mismatch("test") // expected-error {{pragma detect_mismatch is malformed; it requires two comma-separated string literals}}
#pragma detect_mismatch("test", 1) // expected-error {{expected string literal in pragma detect_mismatch}}
#pragma detect_mismatch("test", BAR)

// __pragma

__pragma(comment(linker," bar=" BAR))                                            

#define MACRO_WITH__PRAGMA { \
  __pragma(warning(push)); \
  __pragma(warning(disable: 10000)); \
  1 + (2 > 3) ? 4 : 5; \
  __pragma(warning(pop)); \
}

void f()
{
  __pragma()

  // If we ever actually *support* __pragma(warning(disable: x)),
  // this warning should go away.
  MACRO_WITH__PRAGMA // expected-warning {{lower precedence}} \
                     // expected-note 2 {{place parentheses}}
}


// This should include macro_arg_directive even though the include
// is looking for test.h  This allows us to assign to "n"
#pragma include_alias("test.h", "macro_arg_directive.h" )
#include "test.h"
void test( void ) {
  n = 12;
}

#pragma include_alias(<bar.h>, "bar.h") // expected-warning {{angle-bracketed include <bar.h> cannot be aliased to double-quoted include "bar.h"}}
#pragma include_alias("foo.h", <bar.h>) // expected-warning {{double-quoted include "foo.h" cannot be aliased to angle-bracketed include <bar.h>}}
#pragma include_alias("test.h") // expected-warning {{pragma include_alias expected ','}}

// Make sure that the names match exactly for a replacement, including path information.  If
// this were to fail, we would get a file not found error
#pragma include_alias(".\pp-record.h", "does_not_exist.h")
#include "pp-record.h"

#pragma include_alias(12) // expected-warning {{pragma include_alias expected include filename}}

// It's expected that we can map "bar" and <bar> separately
#define test
// We can't actually look up stdio.h because we're using cc1 without header paths, but this will ensure
// that we get the right bar.h, because the "bar.h" will undef test for us, where <bar.h> won't
#pragma include_alias(<bar.h>, <stdio.h>)
#pragma include_alias("bar.h", "pr2086.h")  // This should #undef test

#include "bar.h"
#if defined(test)
// This should not warn because test should not be defined
#pragma include_alias("test.h")
#endif

// Test to make sure there are no use-after-free problems
#define B "pp-record.h"
#pragma include_alias("quux.h", B)
void g() {}
#include "quux.h"

// Make sure that empty includes don't work
#pragma include_alias("", "foo.h")  // expected-error {{empty filename}}
#pragma include_alias(<foo.h>, <>)  // expected-error {{empty filename}}

// Test that we ignore pragma warning.
#pragma warning(push)
#pragma warning(push, 1)
#pragma warning(disable : 4705)
#pragma warning(disable : 123 456 789 ; error : 321)
#pragma warning(once : 321)
#pragma warning(suppress : 321)
#pragma warning(default : 321)
#pragma warning(pop)

#pragma warning(push, 0)
// FIXME: We could probably support pushing warning level 0.
#pragma warning(pop)

#pragma warning  // expected-warning {{expected '('}}
#pragma warning(   // expected-warning {{expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
#pragma warning()   // expected-warning {{expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
#pragma warning(push 4)  // expected-warning {{expected ')'}}
#pragma warning(push  // expected-warning {{expected ')'}}
#pragma warning(push, 5)  // expected-warning {{requires a level between 0 and 4}}
#pragma warning(pop, 1)  // expected-warning {{expected ')'}}
#pragma warning(push, 1) asdf // expected-warning {{extra tokens at end of #pragma warning directive}}
#pragma warning(disable 4705) // expected-warning {{expected ':'}}
#pragma warning(disable : 0) // expected-warning {{expected a warning number}}
#pragma warning(default 321) // expected-warning {{expected ':'}}
#pragma warning(asdf : 321) // expected-warning {{expected 'push', 'pop'}}
#pragma warning(push, -1) // expected-warning {{requires a level between 0 and 4}}
