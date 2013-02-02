// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -g -emit-llvm %s -o - | FileCheck %s

// CHECK: !{{.*}} = metadata !{i32 {{.*}}, metadata ![[FILE:.*]], metadata !"H", metadata ![[FILE]], i32 6, i64 0, i64 8, i32 0, i32 512, null, metadata !{{.*}}, i32 16, i32 0, i32 0} ; [ DW_TAG_structure_type ]

@interface H
-(void) foo;
@end

@implementation H
-(void) foo {
}
@end

