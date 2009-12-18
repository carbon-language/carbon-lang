// RUN: env RC_DEBUG_OPTIONS=1 %clang -ccc-host-triple i386-apple-darwin9 -g -Os %s  -emit-llvm -S -o - | FileCheck %s
// <rdar://problem/7256886>

// CHECK: !1 = metadata !{
// CHECK: -cc1
// CHECK: -triple i386-apple-darwin9
// CHECK: -g
// CHECK: -Os
// CHECK: [DW_TAG_compile_unit ]

int x;
