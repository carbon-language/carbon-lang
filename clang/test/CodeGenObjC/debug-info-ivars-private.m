// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s

// Debug symbols for private ivars. This test ensures that we are
// generating debug info for ivars added by the implementation.
__attribute((objc_root_class)) @interface NSObject {
  id isa;
}
@end

@protocol Protocol
@end

@interface Delegate : NSObject<Protocol> {
  @protected int foo;
}
@end

@interface Delegate(NSObject)
- (void)f;
@end

@implementation Delegate(NSObject)
- (void)f { return; }
@end

@implementation Delegate {
  int bar;
}

- (void)g:(NSObject*) anObject {
  bar = foo;
}
@end

// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"foo", metadata !{{[0-9]*}}, i32 14, i64 32, i64 32, i64 0, i32 2, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [foo] [line 14, size 32, align 32, offset 0] [protected] [from int]
// CHECK: metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata !"bar", metadata !{{[0-9]*}}, i32 27, i64 32, i64 32, i64 0, i32 1, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [bar] [line 27, size 32, align 32, offset 0] [private] [from int]
