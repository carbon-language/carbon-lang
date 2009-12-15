// RUN: %clang_cc1 -Eonly -verify %s

// Try different path permutations of __has_include with existing file.
#if __has_include("stdio.h")
#else
  #error "__has_include failed (1)."
#endif

#if __has_include(<stdio.h>)
#else
  #error "__has_include failed (2)."
#endif

// Try unary expression.
#if !__has_include("stdio.h")
  #error "__has_include failed (5)."
#endif

// Try binary expression.
#if __has_include("stdio.h") && __has_include("stddef.h")
#else
  #error "__has_include failed (6)."
#endif

// Try non-existing file.
#if __has_include("blahblah.h")
  #error "__has_include failed (7)."
#endif

// Try defined.
#if !defined(__has_include)
  #error "defined(__has_include) failed (8)."
#endif

// Try different path permutations of __has_include_next with existing file.
#if __has_include_next("stddef.h") // expected-warning {{#include_next in primary source file}}
#else
  #error "__has_include failed (1)."
#endif

#if __has_include_next(<stddef.h>) // expected-warning {{#include_next in primary source file}}
#else
  #error "__has_include failed (2)."
#endif

// Try unary expression.
#if !__has_include_next("stdio.h") // expected-warning {{#include_next in primary source file}}
  #error "__has_include_next failed (5)."
#endif

// Try binary expression.
#if __has_include_next("stdio.h") && __has_include("stddef.h") // expected-warning {{#include_next in primary source file}}
#else
  #error "__has_include_next failed (6)."
#endif

// Try non-existing file.
#if __has_include_next("blahblah.h") // expected-warning {{#include_next in primary source file}}
  #error "__has_include_next failed (7)."
#endif

// Try defined.
#if !defined(__has_include_next)
  #error "defined(__has_include_next) failed (8)."
#endif

// Try badly formed expressions.
// FIXME: I don't quite know how to avoid preprocessor side effects.
// Use FileCheck?
// It also assert due to unterminated #if's.
//#if __has_include("stdio.h"
//#if __has_include "stdio.h")
//#if __has_include(stdio.h)
//#if __has_include()
//#if __has_include(
//#if __has_include)
//#if __has_include
//#if __has_include(<stdio.h>
//#if __has_include<stdio.h>)
//#if __has_include("stdio.h)
//#if __has_include(stdio.h")
//#if __has_include(<stdio.h)
//#if __has_include(stdio.h>)
