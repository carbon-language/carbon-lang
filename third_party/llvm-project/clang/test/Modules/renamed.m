@import NewName;

int f() { return same_api; }

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I %S/Inputs/oldname -fmodules-cache-path=%t %s -verify

// expected-no-diagnostics
