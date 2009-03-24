// RUN: clang-cc -emit-llvm %s -o - | grep "align 16" | count 2

__attribute((aligned(16))) float a[128];
union {int a[4]; __attribute((aligned(16))) float b[4];} u;
