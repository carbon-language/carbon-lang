// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/interface1.m
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/interface2.m
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: interface2.m:16:9: error: instance variable 'ivar2' declared with incompatible types in different translation units ('float' vs. 'int')
// CHECK: interface1.m:16:7: note: declared here with type 'int'
// CHECK: interface1.m:21:1: error: class 'I4' has incompatible superclasses
// CHECK: interface1.m:21:17: note: inherits from superclass 'I2' here
// CHECK: interface2.m:21:17: note: inherits from superclass 'I1' here
// CHECK: 5 diagnostics generated

