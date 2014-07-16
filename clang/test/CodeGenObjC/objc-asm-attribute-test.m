// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin %s -o - | FileCheck %s
// rdar://16462586

__attribute__((objc_runtime_name("MySecretNamespace.Protocol")))
@protocol Protocol
- (void) MethodP;
+ (void) ClsMethodP;
@end

__attribute__((objc_runtime_name("MySecretNamespace.Protocol2")))
@protocol Protocol2
- (void) MethodP2;
+ (void) ClsMethodP2;
@end

__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@interface Message <Protocol, Protocol2> {
  id MyIVAR;
}
@end

@implementation Message
- (id) MyMethod {
  return MyIVAR;
}

+ (id) MyClsMethod {
  return 0;
}

- (void) MethodP{}
- (void) MethodP2{}

+ (void) ClsMethodP {}
+ (void) ClsMethodP2 {}
@end

// rdar://16877359
__attribute__((objc_runtime_name("foo")))
@interface SLREarth
- (instancetype)init;
+ (instancetype)alloc;
@end

id Test16877359() {
    return [SLREarth alloc];
}

// CHECK: @"OBJC_IVAR_$_MySecretNamespace.Message.MyIVAR" = global i64
// CHECK: @"OBJC_CLASS_$_MySecretNamespace.Message" = global %struct._class_t
// CHECK: @"OBJC_METACLASS_$_MySecretNamespace.Message" = global %struct._class_t
// CHECK: @"OBJC_CLASS_$_foo" = external global %struct._class_t
// CHECK: define internal i8* @"\01-[Message MyMethod]"
// CHECK: [[IVAR:%.*]] = load i64* @"OBJC_IVAR_$_MySecretNamespace.Message.MyIVAR"
