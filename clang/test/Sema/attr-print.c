// RUN: %clang_cc1 %s -ast-print -fms-extensions | FileCheck %s

// CHECK: int x __attribute__((aligned(4)));
int x __attribute__((aligned(4)));

// FIXME: Print this at a valid location for a __declspec attr.
// CHECK: int y __declspec(align(4));
__declspec(align(4)) int y;

// CHECK: void foo() __attribute__((const));
void foo() __attribute__((const));

// CHECK: void bar() __attribute__((__const));
void bar() __attribute__((__const));

// FIXME: Print these at a valid location for these attributes.
// CHECK: int *p32 __ptr32;
int * __ptr32 p32;

// CHECK: int *p64 __ptr64;
int * __ptr64 p64;
