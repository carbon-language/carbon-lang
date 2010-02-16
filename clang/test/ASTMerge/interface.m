// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/interface1.m
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/interface2.m
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1

// FIXME: FileCheck!

