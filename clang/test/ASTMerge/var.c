// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/var1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/var2.c
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: declared with incompatible types
