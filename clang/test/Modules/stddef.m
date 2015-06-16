@import StdDef.Other;

size_t getSize();

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/StdDef %s -verify
// expected-no-diagnostics
