// RUN: %clang -target x86_64-unknown-unknown -fverbose-asm -g -O0 -S -emit-llvm %s -o - | FileCheck %s
// <rdar://problem/12566646>

class A {
  int x[];
};
A a;

// CHECK: [[ARRAY_TYPE:![0-9]*]]} ; [ DW_TAG_member ] [x]
// CHECK:  !"0x1\00\000\000\0032\000\000\000", null, null, {{![0-9]+}}, [[ELEM_TYPE:![0-9]+]], null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
// CHECK: [[ELEM_TYPE]] = !{[[SUBRANGE:.*]]}
// CHECK: [[SUBRANGE]] = !{!"0x21\000\00-1"} ; [ DW_TAG_subrange_type ] [unbounded]
