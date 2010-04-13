// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -DKEEP_ATTRIBUTES -D"id=struct objc_object *" -D"Class=struct objc_class *" -D"SEL=void*" -D"__declspec(X)=" -emit-llvm -o - %t-rw.cpp | FileCheck %t-rw.cpp
// radar 7738453

void *sel_registerName(const char *);

@interface __NSCFType
@end

@interface __NSCFString : __NSCFType
- (const char *)UTF8String;
@end

@implementation __NSCFString
- (const char *)UTF8String {
    return (const char *)[super UTF8String];
}
@end

// CHECK: call %struct.objc_class* @class_getSuperclass
