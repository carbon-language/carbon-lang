// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin %s -o /dev/null

@interface A
@end
@protocol P
@end
@interface B : A <P>
{
}
@end
@implementation B
- (void)test {
}
@end
