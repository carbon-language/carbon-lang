@import incomplete_mod;

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -Wincomplete-module -fmodules -I %S/Inputs %s 2>&1 | FileCheck %s
// CHECK: {{warning: header '.*incomplete_mod_missing.h' is included in module 'incomplete_mod' but not listed in module map}}
