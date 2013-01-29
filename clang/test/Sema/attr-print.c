// RUN: %clang_cc1 %s -ast-print -fms-extensions | FileCheck %s

// FIXME: we need to fix the "BoolArgument<"IsMSDeclSpec">"
// hack in Attr.td for attribute "Aligned".

// CHECK: int x __attribute__((aligned(4, 0)));
int x __attribute__((aligned(4)));

// FIXME: Print this at a valid location for a __declspec attr.
// CHECK: int y __declspec(align(4, 1));
__declspec(align(4)) int y;

// CHECK: void foo() __attribute__((const));
void foo() __attribute__((const));

// CHECK: void bar() __attribute__((__const));
void bar() __attribute__((__const));
