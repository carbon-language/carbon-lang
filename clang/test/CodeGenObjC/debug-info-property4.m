// RUN: %clang_cc1 -fobjc-default-synthesize-properties -masm-verbose -S -g %s -o - | FileCheck %s

// CHECK: AT_APPLE_property_name
// CHECK: AT_APPLE_property_attribute
// CHECK: AT_APPLE_property


@interface I1
@property int p1;
@end

@implementation I1
@end

void foo(I1 *ptr) {}
