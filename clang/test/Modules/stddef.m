@import StdDef.Other;

size_t getSize();

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -I %S/Inputs/StdDef %s -verify
// expected-no-diagnostics
