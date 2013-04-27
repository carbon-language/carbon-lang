// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -disable-llvm-optzns -o - %s | FileCheck %s
// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -struct-path-tbaa -disable-llvm-optzns -o - %s | FileCheck %s -check-prefix=PATH

// Types with the may_alias attribute should be considered equivalent
// to char for aliasing.

typedef int __attribute__((may_alias)) aliasing_int;

void test0(aliasing_int *ai, int *i)
{
// CHECK: store i32 0, i32* %{{.*}}, !tbaa !1
// PATH: store i32 0, i32* %{{.*}}, !tbaa [[TAG_CHAR:!.*]]
  *ai = 0;
// CHECK: store i32 1, i32* %{{.*}}, !tbaa !3
// PATH: store i32 1, i32* %{{.*}}, !tbaa [[TAG_INT:!.*]]
  *i = 1;
}

// PR9307
struct Test1 { int x; };
struct Test1MA { int x; } __attribute__((may_alias));
void test1(struct Test1MA *p1, struct Test1 *p2) {
  // CHECK: store i32 2, i32* {{%.*}}, !tbaa !1
  // PATH: store i32 2, i32* {{%.*}}, !tbaa [[TAG_CHAR]]
  p1->x = 2;
  // CHECK: store i32 3, i32* {{%.*}}, !tbaa !3
  // PATH: store i32 3, i32* {{%.*}}, !tbaa [[TAG_test1_x:!.*]]
  p2->x = 3;
}

// CHECK: !0 = metadata !{metadata !"any pointer", metadata !1}
// CHECK: !1 = metadata !{metadata !"omnipotent char", metadata !2}
// CHECK: !2 = metadata !{metadata !"Simple C/C++ TBAA"}
// CHECK: !3 = metadata !{metadata !"int", metadata !1}

// PATH: [[TYPE_CHAR:!.*]] = metadata !{metadata !"omnipotent char", metadata !{{.*}}
// PATH: [[TAG_CHAR]] = metadata !{metadata [[TYPE_CHAR]], metadata [[TYPE_CHAR]], i64 0}
// PATH: [[TAG_INT]] = metadata !{metadata [[TYPE_INT:!.*]], metadata [[TYPE_INT]], i64 0}
// PATH: [[TYPE_INT]] = metadata !{metadata !"int", metadata [[TYPE_CHAR]]
// PATH: [[TAG_test1_x]] = metadata !{metadata [[TYPE_test1:!.*]], metadata [[TYPE_INT]], i64 0}
// PATH: [[TYPE_test1]] = metadata !{metadata !"_ZTS5Test1", metadata [[TYPE_INT]], i64 0}
