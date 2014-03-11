// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://16206443

@interface NSObject 
- (void) finalize;
@end

__attribute__((availability(macosx,introduced=9876.5)))
@interface MyClass : NSObject
+ (void)someClassMethod;
- (void)someInstanceMethod;
@end

@implementation MyClass
+ (void)someClassMethod {
}

- (void)someInstanceMethod {
    [MyClass someClassMethod];
    [super finalize];
}
@end

void kit()
{
    MyClass *wrapper = [MyClass alloc];
}

// CHECK: @"OBJC_CLASS_$_MyClass" = global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_MyClass" = global %struct._class_t
// CHECK: @"OBJC_CLASS_$_NSObject" = external global %struct._class_t

