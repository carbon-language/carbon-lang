// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodule-cache-path %t -fmodule-name=macros %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c -verify -fmodule-cache-path %t %s
// RUN: %clang_cc1 -E -fmodules -x objective-c -fmodule-cache-path %t %s | FileCheck -check-prefix CHECK-PREPROCESSED %s
// FIXME: When we have a syntax for modules in C, use that.

@__experimental_modules_import macros;

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
