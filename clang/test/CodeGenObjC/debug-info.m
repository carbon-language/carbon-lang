// RUN: clang-cc -triple i386-apple-darwin9 -g -emit-llvm -o %t %s &&
// RUN: grep -F 'internal constant [8 x i8] c"-[A m0]\00"' %t

@interface A @end
@implementation A
-(void) m0 {}
@end
