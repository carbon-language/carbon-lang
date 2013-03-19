// RUN: %clang -fverbose-asm -g -S -emit-llvm %s -o - | FileCheck %s
@class ForwardObjcClass;
ForwardObjcClass *ptr = 0;

// CHECK: {{.*}} [ DW_TAG_structure_type ] [ForwardObjcClass] [line 2, size 0, align 0, offset 0] [fwd]
