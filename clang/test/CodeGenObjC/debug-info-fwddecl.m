// RUN: %clang -fverbose-asm -g -S -emit-llvm %s -o - | FileCheck %s
@class ForwardObjcClass;
ForwardObjcClass *ptr = 0;

// CHECK: !8 = metadata !{i32 720915, metadata !6, metadata !"ForwardObjcClass", metadata !6, i32 2, i64 0, i64 0, i32 0, i32 4, null, null, i32 16, i32 0} ; [ DW_TAG_structure_type ]
