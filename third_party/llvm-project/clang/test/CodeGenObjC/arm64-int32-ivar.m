// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-ios -emit-llvm  -o - %s | FileCheck %s
// rdar://12617764

// CHECK: @"OBJC_IVAR_$_I.IVAR2" = global i32 8
// CHECK: @"OBJC_IVAR_$_I.IVAR1" = global i32 0
@interface I
{
	id IVAR1;
	id IVAR2;
}
@end

@implementation I
// CHECK: [[IVAR:%.*]] = load i32, i32* @"OBJC_IVAR_$_I.IVAR2"
// CHECK: [[CONV:%.*]] = sext i32 [[IVAR]] to i64
- (id) METH { return IVAR2; }
@end
