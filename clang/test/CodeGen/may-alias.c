// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -no-struct-path-tbaa -disable-llvm-optzns -o - %s | FileCheck %s
// RUN: %clang_cc1 -Werror -triple i386-unknown-unknown -emit-llvm -O1 -disable-llvm-optzns -o - %s | FileCheck %s -check-prefix=PATH

// Types with the may_alias attribute should be considered equivalent
// to char for aliasing.

typedef int __attribute__((may_alias)) aliasing_int;

void test0(aliasing_int *ai, int *i)
{
// CHECK: store i32 0, i32* %{{.*}}, !tbaa [[TAG_CHAR:!.*]]
// PATH: store i32 0, i32* %{{.*}}, !tbaa [[TAG_CHAR:!.*]]
  *ai = 0;
// CHECK: store i32 1, i32* %{{.*}}, !tbaa [[TAG_INT:!.*]]
// PATH: store i32 1, i32* %{{.*}}, !tbaa [[TAG_INT:!.*]]
  *i = 1;
}

// PR9307
struct Test1 { int x; };
struct Test1MA { int x; } __attribute__((may_alias));
void test1(struct Test1MA *p1, struct Test1 *p2) {
  // CHECK: store i32 2, i32* {{%.*}}, !tbaa [[TAG_CHAR]]
  // PATH: store i32 2, i32* {{%.*}}, !tbaa [[TAG_CHAR]]
  p1->x = 2;
  // CHECK: store i32 3, i32* {{%.*}}, !tbaa [[TAG_INT]]
  // PATH: store i32 3, i32* {{%.*}}, !tbaa [[TAG_test1_x:!.*]]
  p2->x = 3;
}
// CHECK: metadata !{metadata !"any pointer", metadata [[TYPE_CHAR:!.*]],
// CHECK: [[TYPE_CHAR]] = metadata !{metadata !"omnipotent char", metadata [[TAG_CXX_TBAA:!.*]],
// CHECK: [[TAG_CXX_TBAA]] = metadata !{metadata !"Simple C/C++ TBAA"}
// CHECK: [[TAG_CHAR]] = metadata !{metadata [[TYPE_CHAR]], metadata [[TYPE_CHAR]], i64 0}
// CHECK: [[TAG_INT]] = metadata !{metadata [[TYPE_INT:!.*]], metadata [[TYPE_INT]], i64 0}
// CHECK: [[TYPE_INT]] = metadata !{metadata !"int", metadata [[TYPE_CHAR]]

// PATH: [[TYPE_CHAR:!.*]] = metadata !{metadata !"omnipotent char", metadata !{{.*}}
// PATH: [[TAG_CHAR]] = metadata !{metadata [[TYPE_CHAR]], metadata [[TYPE_CHAR]], i64 0}
// PATH: [[TAG_INT]] = metadata !{metadata [[TYPE_INT:!.*]], metadata [[TYPE_INT]], i64 0}
// PATH: [[TYPE_INT]] = metadata !{metadata !"int", metadata [[TYPE_CHAR]]
// PATH: [[TAG_test1_x]] = metadata !{metadata [[TYPE_test1:!.*]], metadata [[TYPE_INT]], i64 0}
// PATH: [[TYPE_test1]] = metadata !{metadata !"Test1", metadata [[TYPE_INT]], i64 0}
