// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -F %S/Inputs %s -verify -DA
// FIXME: getCanonicalName() is not implemented on Windows.
// XFAIL: win32

@import MOdule; // expected-error{{module 'MOdule' not found}}
@import Module;
