@import NewName;

int f() { return same_api; }

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -I %S/Inputs -fmodules-cache-path=%t %s -verify

// expected-no-diagnostics
