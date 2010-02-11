// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/struct1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/struct2.c
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: struct2.c:13:11: error: external variable 'x1' declared with incompatible types in different translation units ('struct S1' vs. 'struct S1')
// CHECK: struct1.c:16:11: note: declared here with type 'struct S1'
// CHECK: 2 diagnostics
