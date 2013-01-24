// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodule-cache-path %t -fdisable-module-hash -fmodules -generate-module-index -F %S/Inputs %s -verify
// RUN: ls %t|grep modules.idx
// REQUIRES: shell
// XFAIL: mingw32

// expected-no-diagnostics
@import DependsOnModule;

