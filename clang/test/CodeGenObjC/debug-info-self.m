// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -g %s -o - | FileCheck %s
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

// CHECK: !MDLocalVariable(tag: DW_TAG_arg_variable, name: "self", arg: 1,
// CHECK-SAME:             scope: ![[CTOR:[0-9]+]]
// CHECK-NOT:              line:
// CHECK-SAME:             flags: DIFlagArtificial | DIFlagObjectPointer{{[,)]}}
// CHECK: !MDLocalVariable(tag: DW_TAG_arg_variable, name: "_cmd", arg: 2,
// CHECK-SAME:             scope: ![[CTOR]]
// CHECK-NOT:              line:
// CHECK-SAME:             flags: DIFlagArtificial{{[,)]}}
// CHECK: !MDLocalVariable(tag: DW_TAG_arg_variable, name: "myarg", arg: 3,
// CHECK-SAME:             scope: ![[CTOR]]
// CHECK-SAME:             line: 11
// CHECK-NOT:              flags:
// CHECK-SAME:             ){{$}}
