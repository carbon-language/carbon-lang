// RUN: %clang %s -emit-llvm -S -o - | FileCheck %s
// Verify that clang version appears in the llvm.ident metadata.

// CHECK: !llvm.ident = !{!0}
// CHECK: !0 = metadata !{metadata !{{.*}}

