// RUN: %clang_cc1 -fsyntax-only -verify %s

#if defined(INCLUDE)
// -------
// This section acts like a header file.
// -------

// Check the use of static variables in non-static inline functions.
static int staticVar; // expected-note + {{'staticVar' declared here}}
static int staticFunction(); // expected-note + {{'staticFunction' declared here}}
const int constVar = 0; // no-warning

namespace {
  int anonVar; // expected-note + {{'anonVar' declared here}}
  int anonFunction(); // expected-note + {{'anonFunction' declared here}}
  const int anonConstVar = 0; // no-warning

  class Anon {
  public:
    static int var; // expected-note + {{'var' declared here}}
    static const int constVar = 0; // no-warning
  };
}

inline void useStatic() { // expected-note + {{use 'static' to give inline function 'useStatic' internal linkage}}
  staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline function with external linkage}}
  (void)staticVar; // expected-warning{{variable 'staticVar' has internal linkage but is used in an inline function with external linkage}}
  anonFunction(); // expected-warning{{function 'anonFunction' is in an anonymous namespace but is used in an inline function with external linkage}}
  (void)anonVar; // expected-warning{{variable 'anonVar' is in an anonymous namespace but is used in an inline function with external linkage}}
  (void)Anon::var; // expected-warning{{variable 'var' is in an anonymous namespace but is used in an inline function with external linkage}}

  (void)constVar; // no-warning
  (void)anonConstVar; // no-warning
  (void)Anon::constVar; // no-warning
}

extern inline int useStaticFromExtern() { // no suggestions
  staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline function with external linkage}}
  return staticVar; // expected-warning{{variable 'staticVar' has internal linkage but is used in an inline function with external linkage}}
}

class A {
public:
  static inline int useInClass() { // no suggestions
    return staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline method with external linkage}}
  }
  inline int useInInstance() { // no suggestions
    return staticFunction(); // expected-warning{{function 'staticFunction' has internal linkage but is used in an inline method with external linkage}}
  }
};

static inline void useStaticFromStatic () {
  // No warnings.
  staticFunction();
  (void)staticVar;
  (void)constVar;
  anonFunction();
  (void)anonVar;
  (void)anonConstVar;
  (void)Anon::var;
  (void)Anon::constVar;
}

namespace {
  inline void useStaticFromAnon() {
    // No warnings.
    staticFunction();
    (void)staticVar;
    (void)constVar;
    anonFunction();
    (void)anonVar;
    (void)anonConstVar;
    (void)Anon::var;
    (void)Anon::constVar;
  }
}

#else
// -------
// This is the main source file.
// -------

#define INCLUDE
#include "inline.cpp"

// Check that we don't allow illegal uses of inline
// (checking C++-only constructs here)
struct c {inline int a;}; // expected-error{{'inline' can only appear on functions}}

// Check that the warnings from the "header file" aren't on by default in
// the main source file.

inline int useStaticMainFile () {
  anonFunction(); // no-warning
  return staticVar; // no-warning
}

// Check that the warnings don't show up even when explicitly requested in C++.

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Winternal-linkage-in-inline"

inline int useStaticAgain () {
  anonFunction(); // no-warning
  return staticVar; // no-warning
}

#pragma clang diagnostic pop

#endif
