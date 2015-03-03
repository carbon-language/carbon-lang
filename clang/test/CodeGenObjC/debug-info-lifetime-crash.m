// RUN: %clang_cc1 -triple arm-apple-ios -emit-llvm -g -fblocks -fobjc-runtime=ios-7.0.0 -fobjc-arc %s -o - | FileCheck %s
// rdar://problem/14990656
@protocol NSObject
- (id)copy;
@end
@class W;
@interface View1
@end
@implementation Controller {
    void (^Block)(void);
}
- (void)View:(View1 *)View foo:(W *)W
{
  // The debug type for these two will be identical, because we do not
  // actually emit the ownership qualifier.
  // CHECK: !MDLocalVariable(tag: DW_TAG_auto_variable, name: "weakSelf",
  // CHECK-SAME:             line: [[@LINE+2]]
  // CHECK-SAME:             type: ![[SELFTY:[0-9]+]]
  __attribute__((objc_ownership(weak))) __typeof(self) weakSelf = self;
  Block = [^{
  // CHECK: !MDLocalVariable(tag: DW_TAG_auto_variable, name: "strongSelf",
  // CHECK-SAME:             line: [[@LINE+2]]
  // CHECK-SAME:             type: ![[SELFTY]]
      __attribute__((objc_ownership(strong))) __typeof(self) strongSelf = weakSelf;
    } copy];
}
@end
