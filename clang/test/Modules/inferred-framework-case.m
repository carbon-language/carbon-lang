// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -F %S/Inputs %s -verify -DA
// FIXME: PR20299 - getCanonicalName() is not implemented on Windows.
// REQUIRES: shell

@import MOdule; // expected-error{{module 'MOdule' not found}}
@import Module;
