// RUN: clang-cc -triple i386-apple-darwin9 -g -emit-llvm -o %t %s &&
// RUN: grep '@.str3 = internal constant \[8 x i8\] c"-\[A m0\]\\00"' %t &&
// RUN: grep '@.str4 = internal constant \[9 x i8\] c"\\01-\[A m0\]\\00"' %t &&
// RUN: grep '@llvm.dbg.subprogram = .* @.str3, .* @.str3, .* @.str4,' %t &&
// RUN: grep '@llvm.dbg.composite.* = .* i32 15, i64 0, i64 8, .* i32 16' %t &&
// RUN: true








@interface A @end
@implementation A // Line 15
-(void) m0 {}
@end
