
// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -F %S/Inputs %s -verify
// expected-no-diagnostics

// Note: transitively imports Module.Sub2.
@import Module.Sub;

int getValue() { 
  return *Module_Sub + *Module_Sub2;
}

