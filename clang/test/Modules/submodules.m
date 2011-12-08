
// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodule-cache-path %t -fauto-module-import -F %S/Inputs %s -verify

// Note: transitively imports Module.Sub2.
__import_module__ Module.Sub;

int getValue() { 
  return *Module_Sub + *Module_Sub2;
}

