// RUN: clang-cc -fsyntax-only -verify %s

static int test_NSURLGetResourceValueForKey( id keys )
{
 for ( id key; in keys) {  // expected-error {{parse error}}
  } 
}
