// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs %s 2>&1 | FileCheck %s

__import_module__ MutuallyRecursive1;

// FIXME: Lots of redundant diagnostics here, because the preprocessor
// can't currently tell the parser not to try to load the module again.

// CHECK: MutuallyRecursive2.h:3:19: fatal error: cyclic dependency in module 'MutuallyRecursive1': MutuallyRecursive1 -> MutuallyRecursive2 -> MutuallyRecursive1
// CHECK: MutuallyRecursive1.h:2:19: fatal error: could not build module 'MutuallyRecursive2'
// CHECK: cycles.c:4:19: fatal error: could not build module 'MutuallyRecursive1'

