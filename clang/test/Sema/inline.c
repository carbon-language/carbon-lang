// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

#if defined(INCLUDE)
// -------
// This section acts like a header file.
// -------

// Check the use of static variables in non-static inline functions.
static int staticVar; // expected-note + {{'staticVar' declared here}}
static int staticFunction(); // expected-note + {{'staticFunction' declared here}}

inline int useStatic () { // expected-note 2 {{use 'static' to give inline function 'useStatic' internal linkage}}
  staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{variable 'staticVar' has internal linkage but is used in an inline function with external linkage}}
}

extern inline int useStaticFromExtern () { // no suggestions
  staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{variable 'staticVar' has internal linkage but is used in an inline function with external linkage}}
}

static inline int useStaticFromStatic () {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

#else
// -------
// This is the main source file.
// -------

#define INCLUDE
#include "inline.c"

// Check that we don't allow illegal uses of inline
inline int a; // expected-error{{'inline' can only appear on functions}}
typedef inline int b; // expected-error{{'inline' can only appear on functions}}
int d(inline int a); // expected-error{{'inline' can only appear on functions}}

// Check that the warnings from the "header file" aren't on by default in
// the main source file.

inline int useStaticMainFile () {
  staticFunction(); // no-warning
  return staticVar; // no-warning
}

// Check that the warnings show up when explicitly requested.

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Winternal-linkage-in-inline"

inline int useStaticAgain () { // expected-note 2 {{use 'static' to give inline function 'useStaticAgain' internal linkage}}
  staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{variable 'staticVar' has internal linkage but is used in an inline function with external linkage}}
}

#pragma clang diagnostic pop

#endif


