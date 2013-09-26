// RUN: %clang_cc1 -cc1 -triple arm-apple-ios -emit-llvm -g -fblocks -fobjc-runtime=ios-7.0.0 -fobjc-arc %s -o - | FileCheck %s
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
  // CHECK-DAG: metadata !"weakSelf", metadata !{{[0-9]+}}, i32 [[@LINE+1]], metadata ![[SELFTY:[0-9]+]], i32 0, i32 0, {{.*}}} ; [ DW_TAG_auto_variable ] [weakSelf]
  __attribute__((objc_ownership(weak))) __typeof(self) weakSelf = self;
  Block = [^{
  // CHECK-DAG: metadata !"strongSelf", metadata !{{[0-9]+}}, i32 [[@LINE+1]], metadata ![[SELFTY]], i32 0, i32 0} ; [ DW_TAG_auto_variable ] [strongSelf]
      __attribute__((objc_ownership(strong))) __typeof(self) strongSelf = weakSelf;
    } copy];
}
@end
