// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// self and _cmd are marked as DW_AT_artificial. 
// myarg is not marked as DW_AT_artificial.

@interface MyClass {
}
- (id)init:(int) myarg;
@end

@implementation MyClass
- (id) init:(int) myarg
{
    return self;
}
@end

// CHECK: metadata !{i32 {{.*}}, metadata ![[CTOR:.*]], metadata !"self", null, i32 16777216, metadata !{{.*}}, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [self] [line 0]
// CHECK: metadata !{i32 {{.*}}, metadata ![[CTOR]], metadata !"_cmd", null, i32 33554432, metadata !{{.*}}, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [_cmd] [line 0]
// CHECK: metadata !{i32 {{.*}}, metadata ![[CTOR]], metadata !"myarg", metadata !{{.*}}, i32 50331659, metadata !{{.*}}, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [myarg] [line 11]
