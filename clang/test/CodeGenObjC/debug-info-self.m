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

// It's weird that the first two parameters are recorded as being in a
// different, ("<unknown>") file compared to the third parameter which is 'in'
// the actual source file. (see the metadata node after the arg name in each
// line)
// CHECK: metadata !{i32 {{.*}}, metadata ![[CTOR:.*]], metadata !"self", metadata ![[UNKFILE:.*]], i32 16777227, metadata !{{.*}}, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [self] [line 11]
// CHECK: metadata !{i32 {{.*}}, metadata ![[CTOR]], metadata !"_cmd", metadata ![[UNKFILE]], i32 33554443, metadata !{{.*}}, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [_cmd] [line 11]
// CHECK: metadata !{i32 {{.*}}, metadata ![[CTOR]], metadata !"myarg", metadata !{{.*}}, i32 50331659, metadata !{{.*}}, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [myarg] [line 11]
