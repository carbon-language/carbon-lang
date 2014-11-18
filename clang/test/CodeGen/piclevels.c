// RUN: %clang_cc1 -emit-llvm -pic-level 2 %s -o - | FileCheck %s -check-prefix=CHECK-BIGPIC
// RUN: %clang_cc1 -emit-llvm -pic-level 1 %s -o - | FileCheck %s -check-prefix=CHECK-SMALLPIC

// CHECK-BIGPIC: !llvm.module.flags = !{{{.*}}}
// CHECK-BIGPIC: !{{[0-9]+}} = metadata !{i32 1, metadata !"PIC Level", i32 2}
// CHECK-SMALLPIC: !llvm.module.flags = !{{{.*}}}
// CHECK-SMALLPIC: !{{[0-9]+}} = metadata !{i32 1, metadata !"PIC Level", i32 1}
