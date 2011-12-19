// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

__attribute((aligned(16))) float a[128];
union {int a[4]; __attribute((aligned(16))) float b[4];} b;

// CHECK: @a = {{.*}}zeroinitializer, align 16
// CHECK: @b = {{.*}}zeroinitializer, align 16
