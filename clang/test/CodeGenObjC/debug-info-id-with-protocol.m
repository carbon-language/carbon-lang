// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-default-synthesize-properties -emit-llvm -g %s -o - | FileCheck %s
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
// CHECK: metadata !{i32 {{[0-9]+}}, metadata !{{[0-9]+}}, metadata !"bad_carrier", metadata !{{[0-9]+}}, i32 {{[0-9]+}}, metadata ![[IDTYPE:[0-9]+]], i32 0, i32 0} ; [ DW_TAG_arg_variable ] [bad_carrier] [line 21]
// CHECK: metadata !{i32 {{[0-9]+}}, metadata !{{[0-9]+}}, metadata !"good_carrier", metadata !{{[0-9]+}}, i32 {{[0-9]+}}, metadata !{{.*}}[[IDTYPE]], i32 0, i32 0} ; [ DW_TAG_arg_variable ] [good_carrier] [line 22]
// CHECK !{{.*}}[[IDTYPE]] = metadata !{i32 {{[0-9]+}}, null, metadata !"id", metadata !{{[0-9]+}}, i32 !{{[0-9]+}}, i64 0, i64 0, i64 0, i32 0, metadata !{{[0-9]+}}} ; [ DW_TAG_typedef ] [id]
