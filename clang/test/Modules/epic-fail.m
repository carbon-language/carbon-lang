// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -fmodules -F %S/Inputs -DgetModuleVersion="epic fail" %s 2>&1 | FileCheck %s

@__experimental_modules_import Module;
@__experimental_modules_import DependsOnModule;

// CHECK: error: expected ';' after top level declarator
// CHECK: note: expanded from macro 'getModuleVersion'
// CHECK: fatal error: could not build module 'Module'
// CHECK: fatal error: could not build module 'Module'
// CHECK-NOT: error:
