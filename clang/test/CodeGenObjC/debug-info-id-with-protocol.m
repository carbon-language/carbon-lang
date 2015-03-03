// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s
__attribute((objc_root_class)) @interface NSObject {
	id isa;
}
+ (id)alloc;
- (id)init;
- (id)retain;
@end

void NSLog(id, ...);

@protocol MyProtocol

-(const char *)hello;

@end

@interface MyClass : NSObject {
}

@property (nonatomic, assign) id <MyProtocol> bad_carrier;
@property (nonatomic, assign) id good_carrier;

@end

@implementation MyClass
@end

int main()
{
    @autoreleasepool
    {
        MyClass *my_class = [MyClass alloc];
        NSLog(@"%p\n", my_class.bad_carrier);
        NSLog(@"%p\n", my_class.good_carrier);
    }
}
// Verify that the debug type for both variables is 'id'.
// CHECK:  ![[IDTYPE:[0-9]+]] = !MDDerivedType(tag: DW_TAG_typedef, name: "id"
//
// CHECK:  !MDLocalVariable(tag: DW_TAG_arg_variable, name: "bad_carrier"
// CHECK-NOT:               line:
// CHECK-SAME:              type: ![[IDTYPE]]
//
// CHECK:  !MDLocalVariable(tag: DW_TAG_arg_variable, name: "good_carrier"
// CHECK-NOT:               line:
// CHECK-SAME:              type: ![[IDTYPE]]
