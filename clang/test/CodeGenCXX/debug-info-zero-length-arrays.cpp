// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S -emit-llvm %s -o - | FileCheck %s
// <rdar://problem/12566646>

class A {
  int x[];
};
A a;

// CHECK: metadata [[ARRAY_TYPE:![0-9]*]]} ; [ DW_TAG_member ] [x]
// CHECK: metadata [[ELEM_TYPE:![0-9]*]], i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
// CHECK: [[ELEM_TYPE]] = metadata !{metadata [[SUBRANGE:.*]]}
// CHECK: [[SUBRANGE]] = metadata !{i32 786465, i64 0, i64 -1} ; [ DW_TAG_subrange_type ] [unbounded]
