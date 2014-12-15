// RUN: %clang_cc1 -x objective-c -emit-llvm -g < %s | FileCheck %s
// Test to check that "self" argument is assigned a location.
// CHECK: call void @llvm.dbg.declare(metadata %0** %{{[^,]+}}, metadata [[SELF:![0-9]*]], metadata !{{.*}})
// CHECK: [[SELF]] = {{.*}} ; [ DW_TAG_arg_variable ] [self]

@interface Foo 
-(void) Bar: (int)x ;
@end


@implementation Foo
-(void) Bar: (int)x 
{
}
@end

