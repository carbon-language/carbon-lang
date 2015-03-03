// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S -emit-llvm %s -o - | FileCheck %s
// <rdar://problem/12566646>

class A {
  int x[];
};
A a;

// CHECK: !MDDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:           baseType: [[ARRAY_TYPE:![0-9]+]]
// CHECK: [[ARRAY_TYPE]] = !MDCompositeType(tag: DW_TAG_array_type,
// CHECK-NOT:                               size:
// CHECK-SAME:                              align: 32
// CHECK-SAME:                              elements: [[ELEM_TYPE:![0-9]+]]
// CHECK: [[ELEM_TYPE]] = !{[[SUBRANGE:.*]]}
// CHECK: [[SUBRANGE]] = !MDSubrange(count: -1)
