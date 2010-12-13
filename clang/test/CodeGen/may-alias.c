// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -disable-llvm-optzns -o %t %s
// RUN: FileCheck < %t %s

// Types with the may_alias attribute should be considered equivalent
// to char for aliasing.

typedef int __attribute__((may_alias)) aliasing_int;

void test0(aliasing_int *ai, int *i)
{
  *ai = 0;
  *i = 1;
}

// CHECK: store i32 0, i32* %tmp, !tbaa !1
// CHECK: store i32 1, i32* %tmp1, !tbaa !3

// CHECK: !0 = metadata !{metadata !"any pointer", metadata !1}
// CHECK: !1 = metadata !{metadata !"omnipotent char", metadata !2}
// CHECK: !2 = metadata !{metadata !"Simple C/C++ TBAA", null}
// CHECK: !3 = metadata !{metadata !"int", metadata !1}
