// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fmodules -F %S/Inputs -DgetModuleVersion="epic fail" %s 2>&1 | FileCheck %s

@__experimental_modules_import DependsOnModule;

// CHECK: While building module 'DependsOnModule' imported from
// CHECK: While building module 'Module' imported from
// CHECK: error: expected ';' after top level declarator
// CHECK: note: expanded from macro 'getModuleVersion'
// CHECK: fatal error: could not build module 'Module'
// CHECK: fatal error: could not build module 'DependsOnModule'
// CHECK-NOT: error:
