// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=macros_top %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=macros_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=macros_right %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=macros %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -verify -fmodules-cache-path=%t %s
// RUN: %clang_cc1 -E -fmodules -x objective-c -fmodules-cache-path=%t %s | FileCheck -check-prefix CHECK-PREPROCESSED %s
// FIXME: When we have a syntax for modules in C, use that.
// These notes come from headers in modules, and are bogus.

// FIXME: expected-note@Inputs/macros_left.h:11{{previous definition is here}}
// FIXME: expected-note@Inputs/macros_right.h:12{{previous definition is here}}
// expected-note@Inputs/macros_right.h:12{{expanding this definition of 'LEFT_RIGHT_DIFFERENT'}}
// expected-note@Inputs/macros_top.h:13{{other definition of 'TOP_RIGHT_REDEF'}}
// expected-note@Inputs/macros_right.h:13{{expanding this definition of 'LEFT_RIGHT_DIFFERENT2'}}
// expected-note@Inputs/macros_left.h:14{{other definition of 'LEFT_RIGHT_DIFFERENT'}}
// expected-note@Inputs/macros_right.h:17{{expanding this definition of 'TOP_RIGHT_REDEF'}}

@import macros;

#ifndef INTEGER
#  error INTEGER macro should be visible
#endif

#ifdef FLOAT
#  error FLOAT macro should not be visible
#endif

#ifdef MODULE
#  error MODULE macro should not be visible
#endif

// CHECK-PREPROCESSED: double d
double d;
DOUBLE *dp = &d;

#__public_macro WIBBLE // expected-error{{no macro named 'WIBBLE'}}

void f() {
  // CHECK-PREPROCESSED: int i = INTEGER;
  int i = INTEGER; // the value was exported, the macro was not.
  i += macros; // expanded from __MODULE__ within the 'macros' module.
}

#ifdef __MODULE__
# error Not building a module!
#endif

#if __building_module(macros)
# error Not building a module
#endif

// None of the modules we depend on have been imported, and therefore
// their macros should not be visible.
#ifdef LEFT
#  error LEFT should not be visible
#endif

#ifdef RIGHT
#  error RIGHT should not be visible
#endif

#ifdef TOP
#  error TOP should not be visible
#endif

// Import left module (which also imports top)
@import macros_left;

#ifndef LEFT
#  error LEFT should be visible
#endif

#ifdef RIGHT
#  error RIGHT should not be visible
#endif

#ifndef TOP
#  error TOP should be visible
#endif

#ifndef TOP_LEFT_UNDEF
#  error TOP_LEFT_UNDEF should still be defined
#endif

void test1() {
  int i;
  TOP_RIGHT_REDEF *ip = &i;
}

#define LEFT_RIGHT_DIFFERENT2 double // FIXME: expected-warning{{'LEFT_RIGHT_DIFFERENT2' macro redefined}} \
                                     // expected-note{{other definition of 'LEFT_RIGHT_DIFFERENT2'}}

// Import right module (which also imports top)
@import macros_right;

#undef LEFT_RIGHT_DIFFERENT3

#ifndef LEFT
#  error LEFT should be visible
#endif

#ifndef RIGHT
#  error RIGHT should be visible
#endif

#ifndef TOP
#  error TOP should be visible
#endif

void test2() {
  int i;
  float f;
  double d;
  TOP_RIGHT_REDEF *fp = &f; // expected-warning{{ambiguous expansion of macro 'TOP_RIGHT_REDEF'}}
  
  LEFT_RIGHT_IDENTICAL *ip = &i;
  LEFT_RIGHT_DIFFERENT *ip2 = &i; // expected-warning{{ambiguous expansion of macro 'LEFT_RIGHT_DIFFERENT'}}
  LEFT_RIGHT_DIFFERENT2 *ip3 = &i; // expected-warning{{ambiguous expansion of macro 'LEFT_RIGHT_DIFFERENT2}}
  int LEFT_RIGHT_DIFFERENT3;
}

#define LEFT_RIGHT_DIFFERENT double // FIXME: expected-warning{{'LEFT_RIGHT_DIFFERENT' macro redefined}}

void test3() {
  double d;
  LEFT_RIGHT_DIFFERENT *dp = &d; // okay
  int x = FN_ADD(1,2);
}

#ifndef TOP_RIGHT_UNDEF
#  error TOP_RIGHT_UNDEF should still be defined
#endif

@import macros_right.undef;

#ifndef TOP_RIGHT_UNDEF
# error TOP_RIGHT_UNDEF should still be defined
#endif
