// RUN: %clang_cc1 -emit-llvm -x objective-c %s -o - | FileCheck %s
// rdar://16203115

@interface NSObject @end

@protocol P0;

@interface A : NSObject <P0>
+(Class) getClass;
@end

@implementation A
+(Class) getClass { return self; }
@end

int main() {
  Protocol *P0 = @protocol(P0);
  return 0;
}

// CHECK: @"\01l_OBJC_PROTOCOL_$_P0" = weak hidden global
// CHECK: @"\01l_OBJC_CLASS_PROTOCOLS_$_A" = private global
// CHECK: @"\01l_OBJC_LABEL_PROTOCOL_$_P0" = weak hidden global
// CHECK: @"\01l_OBJC_PROTOCOL_REFERENCE_$_P0" = weak hidden global
