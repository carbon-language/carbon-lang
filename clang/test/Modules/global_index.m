// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodule-cache-path %t -fdisable-module-hash -fmodules -fmodules-global-index -F %S/Inputs %s -verify
// RUN: ls %t|grep modules.idx
// RUN: %clang_cc1 -Wauto-import -fmodule-cache-path %t -fdisable-module-hash -fmodules -fmodules-global-index -F %S/Inputs %s -verify
// REQUIRES: shell

// expected-no-diagnostics
@import DependsOnModule;
@import Module;

int *get_sub() {
  return Module_Sub;
}
