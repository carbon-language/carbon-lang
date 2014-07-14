// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -F %S/Inputs %s -verify -DA

@import MOdule; // expected-error{{module 'MOdule' not found}}
@import Module;
