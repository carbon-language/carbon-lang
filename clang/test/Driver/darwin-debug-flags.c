// RUN: env RC_DEBUG_OPTIONS=1 %clang -ccc-host-triple i386-apple-darwin9 -g -Os %s  -emit-llvm -S -o - | FileCheck %s
// <rdar://problem/7256886>

// CHECK: !1 = metadata !{
// CHECK: -g -Os
// CHECK: -mmacosx-version-min=10.5.0
// CHECK: [ DW_TAG_compile_unit ]

int x;
