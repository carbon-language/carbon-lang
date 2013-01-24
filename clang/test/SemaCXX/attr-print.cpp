// RUN: %clang_cc1 %s -ast-print | FileCheck %s

// FIXME: align attribute print

// CHECK: int x __attribute__((aligned(4, 0)));
int x __attribute__((aligned(4)));

// CHECK: int y __attribute__((align(4, 0)));
int y __attribute__((align(4)));

// CHECK: void foo() __attribute__((const));
void foo() __attribute__((const));

// CHECK: void bar() __attribute__((__const));
void bar() __attribute__((__const));
