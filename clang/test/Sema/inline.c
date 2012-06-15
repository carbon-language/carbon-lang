// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check that we don't allow illegal uses of inline
inline int a; // expected-error{{'inline' can only appear on functions}}
typedef inline int b; // expected-error{{'inline' can only appear on functions}}
int d(inline int a); // expected-error{{'inline' can only appear on functions}}


// Check the use of static variables in non-static inline functions.
static int staticVar; // expected-note 2 {{'staticVar' declared here}}
static int staticFunction(); // expected-note 2 {{'staticFunction' declared here}}

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
