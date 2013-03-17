// RUN: %clang_cc1 -emit-llvm -g -w -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
# 1 "foo.m" 1
# 1 "foo.m" 2
# 1 "./foo.h" 1
@interface NSObject {
  struct objc_object *isa;
}
@end
@class NSDictionary;

@interface Foo : NSObject {}
@property (strong, nonatomic) NSDictionary *dict;
@end
# 2 "foo.m" 2




@implementation Foo
@synthesize dict = _dict;

- (void) bork {
}
@end

int main(int argc, char *argv[]) {
  @autoreleasepool {
    Foo *f = [Foo new];
    [f bork];
  }
}

// CHECK: ![[FILE:.*]] = {{.*}}[ DW_TAG_file_type ] [{{.*}}/foo.h]
// CHECK: !{{.*}} = metadata !{i32 {{.*}}, i32 0, metadata ![[FILE]], metadata !"-[Foo dict]", metadata !"-[Foo dict]", metadata !"", metadata ![[FILE]], i32 8, metadata !{{.*}}, i1 true, i1 true, i32 0, i32 0, null, i32 320, i1 false, %1* (%0*, i8*)* @"\01-[Foo dict]", null, null, metadata !{{.*}}, i32 8} ; [ DW_TAG_subprogram ]
