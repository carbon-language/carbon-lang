// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S -emit-llvm %s -o - | FileCheck %s
// <rdar://problem/12566646>

class A {
  int x[];
};
A a;

// CHECK: !9 = metadata !{i32 {{.*}}, metadata {{.*}}, metadata !"x", metadata {{.*}}, i32 5, i64 0, i64 0, i64 0, i32 1, metadata [[ARRAY_TYPE:.*]]} ; [ DW_TAG_member ]
// CHECK: [[ARRAY_TYPE]] = metadata !{i32 {{.*}}, null, metadata !"", null, i32 0, i64 0, i64 32, i32 0, i32 0, metadata [[BASE_TYPE:.*]], metadata [[ELEM_TYPE:.*]], i32 0, i32 0} ; [ DW_TAG_array_type ]
// CHECK: [[BASE_TYPE]] = metadata !{i32 {{.*}}, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
// CHECK: [[ELEM_TYPE]] = metadata !{metadata [[SUBRANGE:.*]]}
// CHECK: [[SUBRANGE]] = metadata !{i32 786465, i64 0, i64 -1} ; [ DW_TAG_subrange_type ] [unbound]
