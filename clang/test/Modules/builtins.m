@import builtin;

int foo() {
  return __builtin_object_size(p, 0);
}

@import builtin.sub;

int bar() {
  return __builtin_object_size(p, 0);
}


// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs %s -verify
// expected-no-diagnostics
