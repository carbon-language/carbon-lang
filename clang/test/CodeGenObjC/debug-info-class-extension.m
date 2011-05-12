// RUN: %clang_cc1 -fobjc-nonfragile-abi -masm-verbose -S -g %s -o - | FileCheck %s

// CHECK: AT_APPLE_objc_class_extension

@interface I1
@end

@implementation I1 {
int myi2;
}
int myi;
@end

void foo(I1 *iptr) {}

